"""
One command, three claims, an answer for each.

    python quickstart.py

Downloads a 0.5B model on first run (~1 GB, via ModelScope so it works from
mainland China without a VPN), then puts the three headline claims about System
One Models on trial against a model you can actually inspect.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BANNER = r"""
  _            __  __ _       _
 (_) _____   _|  \/  (_)_ __ (_)
 | |/ _ \ \ / / |\/| | | '_ \| |
 | |  __/\ V /| |  | | | | | | |
 |/ \___| \_/ |_|  |_|_|_| |_|_|
|__/
        System One Models, on your own GPU. No API key.
"""


def die(msg: str) -> None:
    print(f"\n[!] {msg}")
    sys.exit(1)


def main() -> None:
    print(BANNER)

    try:
        import torch
    except ImportError:
        die("PyTorch missing. See README for the install command.")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU   : {name}")
        print(f"  VRAM  : {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB")
        if free / 1024**3 < 1.5:
            print("  note  : under 1.5 GB free -- close some GPU apps if loading fails")
    else:
        print("  GPU   : none detected, running on CPU (slower, still works)")
    print(f"  torch : {torch.__version__}")

    model_id = os.environ.get("JEV_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    local = os.environ.get("JEV_MODEL", "")

    if not local:
        print(f"\n  fetching {model_id} (first run only, ~1 GB) ...")
        try:
            from modelscope import snapshot_download

            local = snapshot_download(model_id, cache_dir=str(Path(__file__).parent / ".models"))
        except Exception as e:
            die(f"download failed: {e}\n    Set JEV_MODEL to a local path to skip this.")
    print(f"  model : {Path(local).name}")

    os.environ.setdefault("TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jevmini import Choice, JevMini, Noul, Schema, Score
    from jevmini.calibration import compute_calibration, routing_table
    from jevmini.datasets import CATEGORIES, SUPPORT_TICKETS

    print("\n  loading ...", end="", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        local, dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    print(f" {time.perf_counter()-t0:.1f}s")

    engine = JevMini(model, tok, device=dev)

    schema = Schema(
        instruction="You are a precise customer support ticket classifier.",
        fields=[
            Choice(name="category", options=CATEGORIES,
                   question="Which department should handle this ticket?"),
            Noul(name="urgent", question="Does this ticket require urgent attention?"),
            Noul(name="refund", question="Is the customer asking for a refund?"),
            Score(name="severity", low=1, high=5, question="How severe is this issue?"),
        ],
    )

    ticket = (
        "I've been charged twice for order #4471 and nobody has replied to my "
        "two previous emails. This is now affecting my company's month-end books."
    )

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 70)
    print("CLAIM 1: 'typed probabilistic decisions out' -- no free-text step")
    print("=" * 70)
    print(f"\n  ticket: {ticket[:66]}...\n")

    engine.decide(ticket, schema)  # warm the kernels
    d = engine.decide(ticket, schema)

    for name, f in d.fields.items():
        if f["type"] == "choice":
            top = sorted(f["distribution"].items(), key=lambda x: -x[1])[:3]
            bar = "  ".join(f"{k}:{v:.2f}" for k, v in top)
            print(f"  {name:10} = {str(f['value']):12} conf={f['confidence']:.3f}   {bar}")
        elif f["type"] == "noul":
            print(f"  {name:10} = {str(f['value']):12} p={f['p_true']:.3f}")
        else:
            print(f"  {name:10} = {f['value']:<12} expected={f['expected']:.2f}")

    print(f"\n  {d.latency_ms:.0f} ms, {d.forward_passes} forward passes, "
          f"{d.scored_continuations} options scored, 0 tokens generated.")
    print("  The return value is a dict of typed values, not a string to parse.")

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 70)
    print("CLAIM 2: 'zero hallucinations' -- true, but it is a type guarantee")
    print("=" * 70)

    n = len(SUPPORT_TICKETS)
    c_lat, g_lat, g_err, g_toks = [], [], 0, []
    print(f"\n  running {n} tickets both ways ...", flush=True)
    for it in SUPPORT_TICKETS:
        c_lat.append(engine.decide(it["text"], schema).latency_ms)
    for it in SUPPORT_TICKETS:
        r = engine.generate_json_baseline(it["text"], schema)
        g_lat.append(r["latency_ms"])
        g_toks.append(r["new_tokens"])
        if r["type_error"]:
            g_err += 1

    cm, gm = statistics.median(c_lat), statistics.median(g_lat)
    print(f"\n  {'':20} {'constrained':>13} {'JSON generate':>15}")
    print("  " + "-" * 50)
    print(f"  {'median latency':20} {cm:10.0f} ms {gm:12.0f} ms")
    print(f"  {'tokens generated':20} {0:13} {sum(g_toks)/len(g_toks):15.0f}")
    print(f"  {'type errors':20} {'0 / ' + str(n):>13} {str(g_err) + ' / ' + str(n):>15}")
    print(f"\n  {gm/cm:.1f}x faster on this box.")
    print("  But note WHY the left column is 0: the output is an index into a")
    print("  Python list. It is arithmetic, not a benchmark result -- and it says")
    print("  nothing about whether the answer is correct.")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 70)
    print("CLAIM 3: calibrated confidence -- the one worth actually measuring")
    print("=" * 70)

    rows = []
    for it in SUPPORT_TICKETS:
        f = engine.decide(it["text"], schema).fields["category"]
        rows.append((f["confidence"], f["value"] == it["category"]))

    rep = compute_calibration([r[0] for r in rows], [r[1] for r in rows], n_bins=5)
    print(f"\n  {rep.summary()}\n")
    print(rep.reliability_table())
    print("\n  Routing policy this would support:\n")
    print(routing_table([r[0] for r in rows], [r[1] for r in rows]))

    print("\n" + "=" * 70)
    print("  A 0.5B model is not Jev. That is the point: you now have a local")
    print("  yardstick. Run scripts/benchmark.py for the full report, and see")
    print("  README for what these numbers do and do not establish.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
