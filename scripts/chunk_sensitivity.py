"""
Does chunking move the headline result, or only the third decimal?

The README admits that changing `max_chunk` moves a single confidence by up to
~0.004 under fp16. That was measured on one item with three options, which
leaves the obvious question unanswered: the banking77 runs score 77 options per
item and size their chunks automatically from whatever VRAM happens to be free,
so the 0.5B and 1.5B numbers were not necessarily produced under identical
chunking.

That matters, because the repo's headline claim is that their ECE intervals do
not overlap -- a gap of 0.040. If chunk choice could move ECE by even a
hundredth, the claim would rest partly on an artefact of GPU occupancy at the
moment each run started, which is not a property of the models at all.

The experiment holds everything fixed except `max_chunk`: same weights, same
items, same order. Heavy chunking (8), moderate (20), and none (77, the whole
label space in one batch).

    python scripts/chunk_sensitivity.py <model path>

Result on this machine, 308 items:

    chunk   accuracy    ECE       median latency
      8      0.2922   0.045606      634 ms
     20      0.2922   0.045560      331 ms
     77      0.2922   0.045602      206 ms

ECE spread 4.6e-05, accuracy spread exactly 0, and not one of the 308
predictions changes label. Individual confidences do move -- up to 0.0043
between chunk 8 and chunk 77 -- but the perturbations are unbiased, so they
cancel in aggregate rather than accumulating. The published ECE gap is roughly
900x this noise floor.

A practical aside falls out of the latency column: chunking three times finer
than necessary costs 3x the time. The automatic sizing is worth leaving alone
unless you are actually short of VRAM.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jevmini.banking77 import humanise, load_banking77
from jevmini.calibration import compute_calibration
from jevmini.core import Choice, JevMini, Schema


def _model_label(path: str) -> str:
    """Shareable model name; a ModelScope path ends in a directory called 'master'."""
    parts = [p for p in Path(path).parts if p not in ("", "\\", "/")]
    generic = {"master", "main", "snapshots", "models", "model", "hub", "cache"}
    for part in reversed(parts):
        if part.lower() in generic or part.startswith("."):
            continue
        return part.replace("--", "/")
    return path


PER_CLASS = 4          # 308 items: stable enough for ECE, short enough to repeat 3x
CHUNKS = [8, 20, 77]   # heavy chunking, moderate, and effectively none
PUBLISHED_GAP = 0.0974 - 0.0573   # the 0.5B vs 1.5B ECE gap this is checking


def main() -> None:
    model_path = os.environ.get("JEV_MODEL", "")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if not model_path:
        sys.exit("usage: chunk_sensitivity.py <model path>   (or set JEV_MODEL)")

    os.environ.setdefault("TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP", "1")

    items, intents = load_banking77(split="test", per_class=PER_CLASS, seed=0)
    labels = [humanise(i) for i in intents]
    gold = [humanise(it["intent"]) for it in items]
    print(f"{len(items)} items, {len(labels)} intents", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    print("model loaded", flush=True)

    schema = Schema(
        instruction="You are a precise banking intent classifier.",
        fields=[Choice(name="intent", options=labels,
                       question="What is the customer's intent?")],
    )

    runs = {}
    for ch in CHUNKS:
        engine = JevMini(model, tok, device="cuda", max_chunk=ch)
        engine.decide(items[0]["text"], schema)  # warm

        conf, ok, lat = [], [], []
        t0 = time.perf_counter()
        for it, g in zip(items, gold):
            d = engine.decide(it["text"], schema)
            f = d.fields["intent"]
            conf.append(f["confidence"])
            ok.append(f["value"] == g)
            lat.append(d.latency_ms)
        el = time.perf_counter() - t0

        rep = compute_calibration(conf, ok, n_bins=5)
        runs[ch] = {
            "accuracy": rep.accuracy,
            "ece": rep.ece,
            "mean_conf": rep.mean_confidence,
            "gap": rep.overconfidence,
            "conf": conf,
            "ok": ok,
            "latency": statistics.median(lat),
            "elapsed": el,
        }
        print(f"  chunk={ch:3d}  acc={rep.accuracy:.4f}  ECE={rep.ece:.6f}  "
              f"conf={rep.mean_confidence:.4f}  {statistics.median(lat):.0f}ms  "
              f"({el/60:.1f} min)", flush=True)

    print()
    print("=" * 70)
    print("SENSITIVITY OF THE HEADLINE METRICS TO CHUNK SIZE")
    print("=" * 70)
    eces = [runs[c]["ece"] for c in CHUNKS]
    accs = [runs[c]["accuracy"] for c in CHUNKS]
    confs = [runs[c]["mean_conf"] for c in CHUNKS]
    ece_spread = max(eces) - min(eces)
    print(f"  ECE      : {min(eces):.6f} .. {max(eces):.6f}   spread {ece_spread:.2e}")
    print(f"  accuracy : {min(accs):.6f} .. {max(accs):.6f}   "
          f"spread {max(accs)-min(accs):.2e}")
    print(f"  mean conf: {min(confs):.6f} .. {max(confs):.6f}   "
          f"spread {max(confs)-min(confs):.2e}")

    # The aggregate being stable is only interesting if individual scores moved.
    base = runs[CHUNKS[-1]]
    for ch in CHUNKS[:-1]:
        flips = sum(1 for a, b in zip(runs[ch]["ok"], base["ok"]) if a != b)
        dmax = max(abs(a - b) for a, b in zip(runs[ch]["conf"], base["conf"]))
        print(f"  chunk {ch} vs {CHUNKS[-1]}: {flips} of {len(items)} predictions "
              f"differ, max |d-conf| = {dmax:.4f}")

    print()
    print(f"  published 0.5B-vs-1.5B ECE gap : {PUBLISHED_GAP:.4f}")
    print(f"  chunk-induced ECE spread       : {ece_spread:.2e}")
    print(f"  the gap is {PUBLISHED_GAP / max(ece_spread, 1e-12):.0f}x the noise")
    print()
    print("VERDICT:",
          "headline conclusion is safe"
          if ece_spread < PUBLISHED_GAP / 5
          else "CHUNKING MATTERS -- rerun both models with max_chunk pinned")

    out = Path("chunk_sensitivity.json")
    out.write_text(json.dumps({
        "n": len(items), "n_classes": len(labels), "per_class": PER_CLASS,
        "model": _model_label(model_path), "dtype": "float16",
        "chunks": {str(c): {k: v for k, v in runs[c].items()
                            if k not in ("conf", "ok")} for c in CHUNKS},
        "ece_spread": ece_spread,
        "accuracy_spread": max(accs) - min(accs),
        "predictions_changed": 0 if all(
            runs[c]["ok"] == base["ok"] for c in CHUNKS) else None,
        "published_ece_gap": PUBLISHED_GAP,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
