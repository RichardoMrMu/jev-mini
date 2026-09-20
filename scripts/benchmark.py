"""
The actual experiment.

Runs five things on one GPU and prints numbers rather than adjectives:

  1. Constrained scoring vs. autoregressive JSON, same model, same questions.
  2. Whether K questions really cost about one forward pass.
  3. Calibration of the constrained path -- ECE, Brier, reliability diagram.
  4. Temperature scaling, fitted both ways, to see how much of any calibration
     gap closes with a single fitted scalar.
  5. The routing table those confidences would actually support.

Point 4 is the one that matters for reading the vendor's claims. If fitting a
single number on a handful of examples gets you most of the way, then
calibration is not, by itself, evidence of a new training paradigm.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jevmini.calibration import (
    compute_calibration,
    fit_temperature,
    fit_temperature_cv,
    routing_table,
)
from jevmini.core import Choice, JevMini, Noul, Schema, Score
from jevmini.datasets import (
    CATEGORIES,
    REVIEWS,
    SENTIMENTS,
    SUPPORT_TICKETS,
    split_dev_eval,
)


def _model_label(path: str) -> str:
    """A shareable name for the model, never the caller's directory layout.

    ModelScope checkouts end in a revision directory literally called
    "master", so the last path component alone is useless. Walk up until a
    component looks like a model name, and recover the vendor prefix from the
    "Qwen--Qwen2.5-0.5B-Instruct" form that the cache uses.
    """
    parts = [p for p in Path(path).parts if p not in ("", "\\", "/")]
    generic = {"master", "main", "snapshots", "models", "model", "hub", "cache"}
    for part in reversed(parts):
        if part.lower() in generic or part.startswith("."):
            continue
        return part.replace("--", "/")
    return path


def load(model_path: str, dtype: str = "float16"):
    """Load onto a small, *shared* GPU.

    A 6 GB laptop card running a desktop session has maybe 4 GB genuinely free,
    and two separate things go wrong on the default path:

      * transformers pre-allocates a warmup block sized for the whole model.
        With a browser holding ~2 GB of VRAM that reservation fails even though
        the weights themselves would have fitted.
      * falling back to a CPU load materialises the full state dict in system
        RAM, which on a loaded 16 GB machine hits the pagefile limit
        (OS error 1455) before anything reaches the card.

    So: disable the warmup, load with low_cpu_mem_usage, then move the model.
    """
    print(f"loading {_model_label(model_path)} ...", flush=True)
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Keeps transformers from reserving the whole model up front.
    os.environ.setdefault("TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP", "1")
    if dev == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM free: {free/1024**3:.2f} / {total/1024**3:.2f} GB", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, dtype),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    try:
        model = model.to(dev)
    except torch.OutOfMemoryError:
        # A shared laptop GPU can lose a gigabyte to the desktop between the
        # free-memory check and the copy. Say what is actually wrong and what
        # would fix it, rather than re-raising a CUDA trace at the user.
        free, total = torch.cuda.mem_get_info()
        need = sum(p.numel() * p.element_size() for p in model.parameters())
        raise SystemExit(
            f"\nNot enough free VRAM to load this model.\n"
            f"  model needs ~{need/1024**3:.1f} GB, card has "
            f"{free/1024**3:.1f} GB free of {total/1024**3:.1f} GB\n"
            f"  Close GPU-using apps (browsers are the usual culprit), or run a\n"
            f"  smaller model, or force CPU with CUDA_VISIBLE_DEVICES=''.\n"
        ) from None

    model.eval()

    print(f"  loaded in {time.perf_counter()-t0:.1f}s on {dev}", flush=True)
    if dev == "cuda":
        print(f"  weights: {torch.cuda.memory_allocated()/1024**2:.0f} MB", flush=True)
    return model, tok, dev


def ticket_schema() -> Schema:
    return Schema(
        instruction="You are a precise customer support ticket classifier.",
        fields=[
            Choice(
                name="category",
                options=CATEGORIES,
                question="Which department should handle this ticket?",
            ),
            Noul(name="urgent", question="Does this ticket require urgent attention?"),
            Score(name="severity", low=1, high=5, question="How severe is this issue?"),
        ],
    )


def review_schema() -> Schema:
    return Schema(
        instruction="You are a precise sentiment classifier.",
        fields=[
            Choice(
                name="sentiment",
                options=SENTIMENTS,
                question="What is the overall sentiment of this review?",
            )
        ],
    )


def run_field(engine, items, schema, field_name, gold_key, labels, warmup=True):
    """Score one field across a dataset, collecting everything needed later."""
    if warmup and items:
        engine.decide(items[0]["text"], schema)

    rows = []
    for it in items:
        d = engine.decide(it["text"], schema)
        f = d.fields[field_name]
        gold = it[gold_key]

        if f["type"] == "noul":
            pred, conf = f["value"], f["confidence"]
            dist = [1.0 - f["p_true"], f["p_true"]]
            t_idx = 1 if gold else 0
        else:
            pred, conf = f["value"], f["confidence"]
            dist = [f["distribution"][l] for l in labels]
            t_idx = labels.index(str(gold))

        rows.append(
            {
                "text": it["text"][:70],
                "difficulty": it.get("difficulty", "medium"),
                "gold": gold,
                "pred": pred,
                "correct": str(pred) == str(gold),
                "confidence": conf,
                "dist": dist,
                "true_idx": t_idx,
                "latency_ms": d.latency_ms,
                "prefix_tokens": d.prefix_tokens,
                "scored": d.scored_continuations,
            }
        )
    return rows


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL", ""))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if not args.model:
        sys.exit("pass --model or set JEV_MODEL")

    model, tok, dev = load(args.model, args.dtype)
    engine = JevMini(model, tok, device=dev)

    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    results = {
        "device": gpu,
        # Record the model name, not the caller's directory layout, so the
        # results file can be shared without leaking a local path.
        "model": _model_label(args.model),
        "dtype": args.dtype,
        "torch": torch.__version__,
    }

    t_schema = ticket_schema()

    # ---------------------------------------------------------------- 1
    section("1. CONSTRAINED SCORING vs AUTOREGRESSIVE JSON")
    print("Same model, same 3 questions per ticket. Only the output path differs.\n")

    c_lat, c_err, c_ok = [], 0, 0
    for it in SUPPORT_TICKETS:
        d = engine.decide(it["text"], t_schema)
        c_lat.append(d.latency_ms)
        if d.fields["category"]["value"] == it["category"]:
            c_ok += 1
        # No parse step exists, so no parse can fail.
        if d.fields["category"]["value"] not in CATEGORIES:
            c_err += 1

    g_lat, g_err, g_ok, g_toks = [], 0, 0, []
    for it in SUPPORT_TICKETS:
        r = engine.generate_json_baseline(it["text"], t_schema)
        g_lat.append(r["latency_ms"])
        g_toks.append(r["new_tokens"])
        if r["type_error"]:
            g_err += 1
        elif r["parsed"] and r["parsed"].get("category") == it["category"]:
            g_ok += 1

    n = len(SUPPORT_TICKETS)
    cm, gm = statistics.median(c_lat), statistics.median(g_lat)
    print(f"  {'':22} {'constrained':>14} {'JSON generate':>16}")
    print("  " + "-" * 56)
    print(f"  {'median latency':22} {cm:11.1f} ms {gm:13.1f} ms")
    print(f"  {'mean latency':22} {statistics.mean(c_lat):11.1f} ms {statistics.mean(g_lat):13.1f} ms")
    print(f"  {'speedup (median)':22} {gm/cm:12.2f}x {'1.00x':>16}")
    print(f"  {'output tokens':22} {0:14d} {sum(g_toks)/len(g_toks):13.1f}")
    print(f"  {'type errors':22} {c_err:>10}/{n} {g_err:>13}/{n}")
    print(f"  {'category accuracy':22} {c_ok/n:14.3f} {g_ok/n:16.3f}")

    results["speed"] = {
        "n": n,
        "constrained_median_ms": cm,
        "constrained_mean_ms": statistics.mean(c_lat),
        "generate_median_ms": gm,
        "generate_mean_ms": statistics.mean(g_lat),
        "speedup_median": gm / cm,
        "constrained_type_errors": c_err,
        "generate_type_errors": g_err,
        "constrained_acc": c_ok / n,
        "generate_acc": g_ok / n,
        "generate_mean_new_tokens": sum(g_toks) / len(g_toks),
    }

    # ---------------------------------------------------------------- 2
    section("2. PARALLEL FIELDS: does K questions really cost ~1 pass?")
    print("Growing the schema while holding the input fixed.\n")

    probe = SUPPORT_TICKETS[6]["text"]
    scaling = []
    for k in (1, 2, 3, 5, 8):
        fs = [Choice(name="category", options=CATEGORIES, question="Which department?")]
        extra = [
            Noul(name="urgent", question="Is this urgent?"),
            Score(name="severity", low=1, high=5, question="How severe?"),
            Noul(name="refund", question="Is a refund requested?"),
            Noul(name="churn", question="Is the customer at risk of leaving?"),
            Noul(name="human", question="Does this need a human reviewer?"),
            Choice(name="tone", options=["angry", "neutral", "polite"], question="What is the tone?"),
            Score(name="clarity", low=1, high=5, question="How clearly written is it?"),
        ]
        fs.extend(extra[: k - 1])
        s = Schema(fields=fs)
        engine.decide(probe, s)  # warm
        lats = [engine.decide(probe, s).latency_ms for _ in range(5)]
        d = engine.decide(probe, s)
        med = statistics.median(lats)
        scaling.append({"k": k, "median_ms": med, "scored": d.scored_continuations})
        print(f"  {k} field(s): {med:7.1f} ms   ({d.scored_continuations} continuations scored, "
              f"{d.forward_passes} forward passes)")

    base = scaling[0]["median_ms"]
    print(f"\n  8 fields cost {scaling[-1]['median_ms']/base:.2f}x the latency of 1 field.")
    print("  Autoregressive JSON would need a separate decode for each answer.")
    results["scaling"] = scaling

    # ---------------------------------------------------------------- 3
    section("3. CALIBRATION -- the claim nobody has independently verified")

    all_rows = {}
    for tag, items, schema, fname, gkey, labels in [
        ("ticket_category", SUPPORT_TICKETS, t_schema, "category", "category", CATEGORIES),
        ("ticket_urgent", SUPPORT_TICKETS, t_schema, "urgent", "urgent", ["no", "yes"]),
        ("review_sentiment", REVIEWS, review_schema(), "sentiment", "sentiment", SENTIMENTS),
    ]:
        rows = run_field(engine, items, schema, fname, gkey, labels)
        all_rows[tag] = rows
        rep = compute_calibration(
            [r["confidence"] for r in rows],
            [r["correct"] for r in rows],
            probs_full=[r["dist"] for r in rows],
            true_idx=[r["true_idx"] for r in rows],
            n_bins=5,
        )
        print(f"\n  --- {tag} ---")
        print("  " + rep.summary())
        print(rep.reliability_table())
        results.setdefault("calibration", {})[tag] = {
            "accuracy": rep.accuracy,
            "mean_confidence": rep.mean_confidence,
            "ece": rep.ece,
            "mce": rep.mce,
            "brier": rep.brier,
            "nll": rep.nll,
            "overconfidence": rep.overconfidence,
        }

    print("\n  Accuracy by difficulty (does confidence track difficulty?):")
    for tag, rows in all_rows.items():
        print(f"\n  {tag}:")
        for diff in ("easy", "medium", "hard"):
            sub = [r for r in rows if r["difficulty"] == diff]
            if not sub:
                continue
            acc = sum(1 for r in sub if r["correct"]) / len(sub)
            conf = sum(r["confidence"] for r in sub) / len(sub)
            print(f"    {diff:7} n={len(sub):3d}  acc={acc:.3f}  conf={conf:.3f}  gap={conf-acc:+.3f}")

    # ---------------------------------------------------------------- 4
    section("4. TEMPERATURE SCALING -- how much closes with one fitted scalar?")

    import math

    def logits_of(rows):
        # Recover pre-softmax scores up to a constant; adequate for fitting T.
        return [[math.log(max(p, 1e-12)) for p in r["dist"]] for r in rows]

    def apply_T(dist, T):
        lg = [math.log(max(p, 1e-12)) / T for p in dist]
        m = max(lg)
        ex = [math.exp(v - m) for v in lg]
        s = sum(ex)
        return [e / s for e in ex]

    cat_rows = all_rows["ticket_category"]

    # Cross-validated fit on the full set. The fold-to-fold spread is reported
    # because it is the honest measure of whether this dataset can support the
    # claim at all -- a wide spread means the fitted T is noise.
    T_cv, T_std = fit_temperature_cv(
        logits_of(cat_rows), [r["true_idx"] for r in cat_rows], n_folds=5
    )
    print(f"  5-fold fitted T = {T_cv:.3f} +/- {T_std:.3f}")
    if T_std > 0.15 * max(T_cv, 1e-6):
        print(f"  NOTE: spread across folds is large relative to the estimate.")
        print(f"        With n={len(cat_rows)} the fitted temperature is not reliably")
        print(f"        distinguishable from T=1. Treat what follows as illustrative.")

    # Also do the plain single-split version, since that is what most write-ups
    # report, and the two disagreeing is itself informative.
    dev_items, eval_items = split_dev_eval(SUPPORT_TICKETS, dev_ratio=0.35, seed=0)
    dev_rows = run_field(engine, dev_items, t_schema, "category", "category", CATEGORIES)
    ev_rows = run_field(engine, eval_items, t_schema, "category", "category", CATEGORIES)
    T_single = fit_temperature(logits_of(dev_rows), [r["true_idx"] for r in dev_rows])
    print(f"\n  single split (dev n={len(dev_items)}, eval n={len(eval_items)}): T = {T_single:.3f}")

    before = compute_calibration(
        [r["confidence"] for r in ev_rows],
        [r["correct"] for r in ev_rows],
        probs_full=[r["dist"] for r in ev_rows],
        true_idx=[r["true_idx"] for r in ev_rows],
        n_bins=5,
    )

    rows_out = []
    for tag, T in (("single-split", T_single), ("5-fold CV", T_cv)):
        scaled = [apply_T(r["dist"], T) for r in ev_rows]
        after = compute_calibration(
            [max(d) for d in scaled],
            [r["correct"] for r in ev_rows],
            probs_full=scaled,
            true_idx=[r["true_idx"] for r in ev_rows],
            n_bins=5,
        )
        delta = before.ece - after.ece
        verdict = "improved" if delta > 0 else "WORSE"
        rows_out.append((tag, T, after.ece, delta, verdict))

    print(f"\n  baseline (T=1) on eval: ECE={before.ece:.4f}  Brier={before.brier:.4f}  acc={before.accuracy:.3f}")
    print(f"\n  {'method':14} {'T':>7} {'ECE after':>11} {'delta':>9}   verdict")
    print("  " + "-" * 56)
    for tag, T, ece, delta, verdict in rows_out:
        print(f"  {tag:14} {T:7.3f} {ece:11.4f} {delta:+9.4f}   {verdict}")

    print("\n  Accuracy is unchanged by construction: temperature cannot move an argmax.")
    print("  A negative delta would be a real result rather than a bug: fitting a")
    print("  calibration parameter on too few examples can make calibration worse.")

    results["temperature"] = {
        "T_cv": T_cv,
        "T_cv_std": T_std,
        "T_single_split": T_single,
        "dev_n": len(dev_items),
        "eval_n": len(eval_items),
        "ece_baseline": before.ece,
        "brier_baseline": before.brier,
        "acc": before.accuracy,
        "variants": [
            {"method": t, "T": T, "ece_after": e, "delta": d, "verdict": v}
            for t, T, e, d, v in rows_out
        ],
    }

    # ---------------------------------------------------------------- 5
    section("5. ROUTING -- what the confidences buy you operationally")
    rows = all_rows["ticket_category"]
    print(routing_table([r["confidence"] for r in rows], [r["correct"] for r in rows]))
    print("\n  This table is the reason calibration matters: it is the contract")
    print("  between the model's stated confidence and your automation policy.")

    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n  peak VRAM during run: {peak:.0f} MB")
        results["peak_vram_mb"] = peak

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
