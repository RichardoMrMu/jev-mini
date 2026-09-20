"""
Cross-validation on public intent benchmarks, with resampled error bars.

Two questions, one script:

  A. Does the larger model stay worse calibrated than the smaller one?
  B. Does that hold once the comparison carries a confidence interval?

B is the reason this script replaced a simpler one. Reporting "ECE 0.053 vs
0.114" invites the reading that the gap is settled, and on a few hundred items
it may not be. Every headline number here comes with a bootstrap interval, and
the two models are compared directly rather than by eye.

Datasets, selected to vary one thing -- the size of the label space:

  banking77   77 intents, random baseline 1.3%
  clinc150   150 intents, random baseline 0.67%

Run:
    python scripts/cross_validate.py --dataset banking77 --model <path> --per-class 40
    python scripts/cross_validate.py --dataset clinc150 --model <path> --per-class 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jevmini.calibration import (
    bootstrap_ci,
    compute_calibration,
    fit_temperature_cv,
    routing_table,
)
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


def load_model(model_path: str, dtype: str = "float16"):
    print(f"loading {_model_label(model_path)} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP", "1")
    if dev == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM free: {free/1024**3:.2f} / {total/1024**3:.2f} GB", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, dtype), trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # A shared desktop GPU can lose a gigabyte between the free-memory check
    # and the copy -- a browser repainting is enough, and nvidia-smi will still
    # show plenty free afterwards. That contention is transient, so retry
    # before concluding the card is genuinely too small.
    need = sum(p.numel() * p.element_size() for p in model.parameters())
    for attempt in range(4):
        try:
            model = model.to(dev)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            if attempt == 3:
                raise SystemExit(
                    f"\nNot enough free VRAM after 4 attempts.\n"
                    f"  model needs ~{need/1024**3:.1f} GB, "
                    f"{free/1024**3:.1f} GB free of {total/1024**3:.1f} GB\n"
                    f"  Close GPU apps, use a smaller model, or set "
                    f"CUDA_VISIBLE_DEVICES=''.\n"
                ) from None
            print(f"  VRAM contended ({free/1024**3:.1f} GB free, "
                  f"need {need/1024**3:.1f} GB); retrying in 5s", flush=True)
            time.sleep(5)
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s on {dev}", flush=True)
    return model, tok, dev


def get_dataset(name: str, per_class: int | None):
    if name == "banking77":
        from jevmini.banking77 import humanise, load_banking77

        items, intents = load_banking77(split="test", per_class=per_class, seed=0)
        return items, intents, humanise, "You are a precise banking intent classifier."
    if name == "clinc150":
        from jevmini.clinc150 import humanise, load_clinc150

        items, intents = load_clinc150(split="test", per_class=per_class, seed=0)
        return items, intents, humanise, "You are a precise intent classifier."
    raise SystemExit(f"unknown dataset {name!r}; expected banking77 or clinc150")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=["banking77", "clinc150"])
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", ""))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--per-class", type=int, default=None,
                    help="examples per intent; omit for the whole test split")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.model:
        sys.exit("pass --model or set JEV_MODEL")

    items, intents, humanise, instruction = get_dataset(args.dataset, args.per_class)
    labels = [humanise(i) for i in intents]
    gold = [humanise(it["intent"]) for it in items]

    model, tok, dev = load_model(args.model, args.dtype)
    engine = JevMini(model, tok, device=dev)

    print(f"\n{args.dataset} test: {len(items)} items across {len(labels)} intents")
    print(f"random baseline: {1/len(labels):.4f}\n")

    schema = Schema(
        instruction=instruction,
        fields=[Choice(name="intent", options=labels,
                       question="What is the customer's intent?")],
    )

    engine.decide(items[0]["text"], schema)  # warm

    rows, lat = [], []
    t0 = time.perf_counter()
    for i, (it, g) in enumerate(zip(items, gold), 1):
        d = engine.decide(it["text"], schema)
        f = d.fields["intent"]
        lat.append(d.latency_ms)
        rows.append({
            "correct": f["value"] == g,
            "confidence": f["confidence"],
            "dist": [f["distribution"][l] for l in labels],
            "true_idx": labels.index(g),
        })
        if i % 250 == 0:
            rate = i / (time.perf_counter() - t0)
            print(f"  {i}/{len(items)}  ({rate:.1f}/s, "
                  f"eta {(len(items)-i)/rate/60:.1f} min)", flush=True)
    elapsed = time.perf_counter() - t0

    conf = [r["confidence"] for r in rows]
    ok = [r["correct"] for r in rows]

    rep = compute_calibration(
        conf, ok, probs_full=[r["dist"] for r in rows],
        true_idx=[r["true_idx"] for r in rows], n_bins=5,
    )
    boot = bootstrap_ci(conf, ok, n_boot=args.n_boot, n_bins=5)

    print("\n" + "=" * 72)
    print(f"CALIBRATION on {args.dataset} ({len(labels)}-way)")
    print("=" * 72)
    print("  " + rep.summary())
    print(f"\n  accuracy {boot['accuracy']:.4f}  95% CI "
          f"[{boot['accuracy_ci'][0]:.4f}, {boot['accuracy_ci'][1]:.4f}]")
    print(f"  ECE      {boot['ece']:.4f}  95% CI "
          f"[{boot['ece_ci'][0]:.4f}, {boot['ece_ci'][1]:.4f}]"
          f"   ({args.n_boot} resamples)")
    print()
    print(rep.reliability_table())
    print(f"\n  median latency {statistics.median(lat):.0f} ms  "
          f"({len(items)} items in {elapsed/60:.1f} min)")

    # ---- temperature -----------------------------------------------------
    def logits_of(rs):
        return [[math.log(max(p, 1e-12)) for p in r["dist"]] for r in rs]

    def apply_T(dist, T):
        lg = [math.log(max(p, 1e-12)) / T for p in dist]
        m = max(lg)
        ex = [math.exp(v - m) for v in lg]
        s = sum(ex)
        return [e / s for e in ex]

    T_cv, T_std = fit_temperature_cv(
        logits_of(rows), [r["true_idx"] for r in rows], n_folds=5
    )
    scaled = [apply_T(r["dist"], T_cv) for r in rows]
    after = compute_calibration(
        [max(d) for d in scaled], ok,
        probs_full=scaled, true_idx=[r["true_idx"] for r in rows], n_bins=5,
    )
    boot_after = bootstrap_ci([max(d) for d in scaled], ok,
                              n_boot=args.n_boot, n_bins=5)

    print("\n" + "=" * 72)
    print("TEMPERATURE SCALING")
    print("=" * 72)
    print(f"  5-fold T = {T_cv:.3f} +/- {T_std:.3f}")
    print(f"  ECE  {rep.ece:.4f} -> {after.ece:.4f}   "
          f"({100*(rep.ece-after.ece)/max(rep.ece,1e-9):+.1f}%)")
    print(f"       after 95% CI [{boot_after['ece_ci'][0]:.4f}, "
          f"{boot_after['ece_ci'][1]:.4f}]")
    print(f"  Brier {rep.brier:.4f} -> {after.brier:.4f}")
    print(f"  accuracy unchanged at {after.accuracy:.4f}")

    print("\n" + "=" * 72)
    print("ROUTING")
    print("=" * 72)
    print(routing_table(conf, ok))

    out = {
        "dataset": args.dataset,
        "split": "test",
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "model": _model_label(args.model),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "n": len(items),
        "n_classes": len(labels),
        "per_class": args.per_class,
        "random_baseline": 1 / len(labels),
        "accuracy": boot["accuracy"],
        "accuracy_ci": list(boot["accuracy_ci"]),
        "mean_confidence": rep.mean_confidence,
        "overconfidence": rep.overconfidence,
        "ece": boot["ece"],
        "ece_ci": list(boot["ece_ci"]),
        "mce": rep.mce,
        "brier": rep.brier,
        "nll": rep.nll,
        "n_boot": args.n_boot,
        "median_latency_ms": statistics.median(lat),
        "elapsed_min": elapsed / 60,
        "temperature": {
            "T_cv": T_cv,
            "T_cv_std": T_std,
            "ece_before": rep.ece,
            "ece_after": after.ece,
            "ece_after_ci": list(boot_after["ece_ci"]),
            "brier_before": rep.brier,
            "brier_after": after.brier,
        },
        "bins": rep.bins,
        # Kept so two runs can be compared with calibration.compare_ece
        # without re-running either of them.
        "ece_boot": boot["ece_boot"],
    }
    if dev == "cuda":
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n  peak VRAM: {out['peak_vram_mb']:.0f} MB")

    dest = args.out or f"{args.dataset}_{_model_label(args.model).split('/')[-1]}.json"
    Path(dest).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
