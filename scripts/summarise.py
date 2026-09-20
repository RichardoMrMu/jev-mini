"""
Read every result file and answer the one question they were run to settle:
does the larger model stay worse calibrated, and is the gap real?

Loads nothing onto the GPU -- it only reads JSON, so it is safe to run while an
evaluation is in progress.

    python scripts/summarise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevmini.calibration import compare_ece

ROOT = Path(__file__).resolve().parent.parent


def load_all() -> list[dict]:
    out = []
    for f in sorted(ROOT.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "ece" not in d:
            continue
        # Older runs wrote "banking77 (test)" where the current script writes
        # "banking77". Runs are paired by this string, so a stale file lands
        # under a dataset of its own and gets mispaired against a run of a
        # different size -- which once reported a confident, wrong 1.98x
        # ratio between two models measured on 3080 and 385 items. Normalise
        # on read instead of trusting the files to agree.
        if "dataset" in d:
            d["dataset"] = d["dataset"].split(" ")[0]
        d["_file"] = f.name
        out.append(d)
    return out


def fmt_ci(d: dict, key: str) -> str:
    ci = d.get(f"{key}_ci")
    if not ci:
        return " " * 18
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def main() -> None:
    runs = load_all()

    print("=" * 78)
    print("ALL RUNS")
    print("=" * 78)
    print(f"{'file':24} {'dataset':12} {'n':>5} {'cls':>4} "
          f"{'acc':>6} {'ECE':>7} {'ECE 95% CI':>18} {'dir':>5}")
    print("-" * 78)
    for d in runs:
        ds = d.get("dataset", "hand-built")
        direction = "under" if d.get("overconfidence", 0) < 0 else "over"
        print(f"{d['_file']:24} {ds:12} {d.get('n', 0):5} "
              f"{d.get('n_classes', 5):4} {d['accuracy']:6.3f} {d['ece']:7.4f} "
              f"{fmt_ci(d, 'ece'):>18} {direction:>5}")

    # Pair the two model sizes within each dataset at the largest n available.
    print()
    print("=" * 78)
    print("DOES THE LARGER MODEL STAY WORSE CALIBRATED?")
    print("=" * 78)

    by_ds: dict[str, dict[str, dict]] = {}
    for d in runs:
        ds = d.get("dataset", "hand-built")
        size = "1.5B" if "1.5B" in d.get("model", "") else "0.5B"
        prev = by_ds.setdefault(ds, {}).get(size)
        if prev is None or d.get("n", 0) > prev.get("n", 0):
            by_ds[ds][size] = d

    for ds, pair in by_ds.items():
        if len(pair) < 2:
            continue
        small, large = pair["0.5B"], pair["1.5B"]
        print(f"\n--- {ds}  (n={small.get('n')}, {small.get('n_classes', 5)} classes) ---")
        print(f"  accuracy : 0.5B {small['accuracy']:.4f}  ->  "
              f"1.5B {large['accuracy']:.4f}"
              f"   ({'+' if large['accuracy'] > small['accuracy'] else ''}"
              f"{large['accuracy'] - small['accuracy']:.4f})")
        print(f"  ECE      : 0.5B {small['ece']:.4f}  ->  1.5B {large['ece']:.4f}"
              f"   ({large['ece'] / max(small['ece'], 1e-9):.2f}x)")

        if "ece_boot" in small and "ece_boot" in large:
            print(compare_ece(small, large, "0.5B", "1.5B"))
        elif small.get("ece_ci") and large.get("ece_ci"):
            a_lo, a_hi = small["ece_ci"]
            b_lo, b_hi = large["ece_ci"]
            overlap = not (a_hi < b_lo or b_hi < a_lo)
            print(f"  0.5B 95% CI [{a_lo:.4f}, {a_hi:.4f}]")
            print(f"  1.5B 95% CI [{b_lo:.4f}, {b_hi:.4f}]")
            print("  " + ("intervals overlap -- gap is NOT established"
                          if overlap else
                          "intervals are disjoint -- the gap is real at this n"))
        else:
            print("  (no intervals on one side; run backfill_ci.py)")

    # The label-space hypothesis: does the error direction track class count?
    print()
    print("=" * 78)
    print("DOES THE ERROR DIRECTION TRACK LABEL-SPACE SIZE?")
    print("=" * 78)
    print(f"{'dataset':12} {'classes':>8} {'model':>6} {'signed gap':>12}  direction")
    print("-" * 60)
    rows = []
    for ds, pair in by_ds.items():
        for size, d in sorted(pair.items()):
            rows.append((d.get("n_classes", 5), ds, size, d.get("overconfidence", 0.0)))
    for cls, ds, size, gap in sorted(rows):
        print(f"{ds:12} {cls:8} {size:>6} {gap:+12.4f}  "
              f"{'under-confident' if gap < 0 else 'over-confident'}")

    print()
    print("Read the signed gap as mean_confidence - accuracy: positive means the")
    print("model claims more certainty than it earns, negative means less.")


if __name__ == "__main__":
    main()
