"""
Cross-validation on banking77: do the hand-built findings survive public data?

The hand-labelled set has a structural weakness -- whoever wrote the labels
also wrote the conclusions. This script re-tests the two claims that matter on
a standard benchmark nobody here authored:

  A. Does the larger model stay *worse calibrated* despite being more accurate?
  B. Does a single fitted temperature still recover most of the gap?

banking77 raises the bar in a useful way: 77 classes rather than 5, so random
guessing is 1.3% and there is far more room for probability mass to land in the
wrong place. It is also exactly balanced, so no class prior can flatter the
accuracy.

Run:
    python scripts/cross_validate_banking77.py --model <path> --out b77_0.5b.json
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

from jevmini.banking77 import humanise, load_banking77
from jevmini.calibration import compute_calibration, fit_temperature_cv, routing_table
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


def load(model_path: str, dtype: str = "float16"):
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
    try:
        model = model.to(dev)
    except torch.OutOfMemoryError:
        free, total = torch.cuda.mem_get_info()
        need = sum(p.numel() * p.element_size() for p in model.parameters())
        raise SystemExit(
            f"\nNot enough free VRAM.\n  model needs ~{need/1024**3:.1f} GB, "
            f"{free/1024**3:.1f} GB free of {total/1024**3:.1f} GB\n"
            f"  Close GPU apps, use a smaller model, or set CUDA_VISIBLE_DEVICES=''.\n"
        ) from None
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s on {dev}", flush=True)
    return model, tok, dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", ""))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--per-class", type=int, default=3,
                    help="examples sampled per intent (77 * N total)")
    ap.add_argument("--out", default="b77_results.json")
    args = ap.parse_args()

    if not args.model:
        sys.exit("pass --model or set JEV_MODEL")

    model, tok, dev = load(args.model, args.dtype)
    engine = JevMini(model, tok, device=dev)

    items, intents = load_banking77(split="test", per_class=args.per_class, seed=0)
    labels = [humanise(i) for i in intents]
    gold = [humanise(it["intent"]) for it in items]

    print(f"\nbanking77 test: {len(items)} items across {len(labels)} intents "
          f"({args.per_class} per class)")
    print(f"random baseline: {1/len(labels):.3f}\n")

    schema = Schema(
        instruction="You are a precise banking intent classifier.",
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
            "pred": f["value"],
            "gold": g,
            "correct": f["value"] == g,
            "confidence": f["confidence"],
            "dist": [f["distribution"][l] for l in labels],
            "true_idx": labels.index(g),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(items)} ...", flush=True)
    elapsed = time.perf_counter() - t0

    acc = sum(r["correct"] for r in rows) / len(rows)
    rep = compute_calibration(
        [r["confidence"] for r in rows], [r["correct"] for r in rows],
        probs_full=[r["dist"] for r in rows], true_idx=[r["true_idx"] for r in rows],
        n_bins=5,
    )

    print("\n" + "=" * 72)
    print("CALIBRATION on banking77 (77-way)")
    print("=" * 72)
    print("  " + rep.summary())
    print(rep.reliability_table())
    print(f"\n  median latency {statistics.median(lat):.0f} ms  "
          f"({len(items)} items in {elapsed:.0f}s)")
    print(f"  accuracy {acc:.3f} vs {1/len(labels):.3f} random "
          f"({acc*len(labels):.0f}x baseline)")

    # ---- temperature, cross-validated on this dataset only ----
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
        [max(d) for d in scaled], [r["correct"] for r in rows],
        probs_full=scaled, true_idx=[r["true_idx"] for r in rows], n_bins=5,
    )

    print("\n" + "=" * 72)
    print("TEMPERATURE SCALING")
    print("=" * 72)
    print(f"  5-fold T = {T_cv:.3f} +/- {T_std:.3f}")
    print(f"  ECE  {rep.ece:.4f} -> {after.ece:.4f}   "
          f"({100*(rep.ece-after.ece)/max(rep.ece,1e-9):+.1f}%)")
    print(f"  Brier {rep.brier:.4f} -> {after.brier:.4f}")
    print(f"  accuracy unchanged at {after.accuracy:.3f} (temperature cannot move argmax)")

    print("\n" + "=" * 72)
    print("ROUTING")
    print("=" * 72)
    print(routing_table([r["confidence"] for r in rows], [r["correct"] for r in rows]))

    out = {
        "dataset": "banking77 (test)",
        "source": "ModelScope modelscope/banking77, CC-BY-4.0",
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "model": _model_label(args.model),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "n": len(items),
        "n_classes": len(labels),
        "per_class": args.per_class,
        "random_baseline": 1 / len(labels),
        "accuracy": acc,
        "mean_confidence": rep.mean_confidence,
        "overconfidence": rep.overconfidence,
        "ece": rep.ece,
        "mce": rep.mce,
        "brier": rep.brier,
        "nll": rep.nll,
        "median_latency_ms": statistics.median(lat),
        "temperature": {
            "T_cv": T_cv,
            "T_cv_std": T_std,
            "ece_before": rep.ece,
            "ece_after": after.ece,
            "brier_before": rep.brier,
            "brier_after": after.brier,
        },
        "bins": rep.bins,
    }
    if dev == "cuda":
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n  peak VRAM: {out['peak_vram_mb']:.0f} MB")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
