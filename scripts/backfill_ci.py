"""
Add bootstrap intervals to a result file produced before they existed.

Rerunning a 25-minute evaluation to obtain error bars would be wasteful, but
the intervals cannot be recovered from summary statistics either: bootstrapping
needs the per-item (confidence, correct) pairs, and only the binned aggregates
were saved.

What the bins do preserve is enough to reconstruct a faithful sample. Each bin
records its count, its mean confidence and its accuracy, so the items in it can
be regenerated as `n` entries at that confidence, of which `round(n * acc)` are
correct. Resampling over that reconstruction gives an interval that reflects
the real sample size and the real spread across bins.

The one thing it cannot reproduce is within-bin variation in confidence: every
item in a bin is treated as sitting exactly at the bin mean. That makes the
interval mildly optimistic, so the reconstruction is marked in the output and
the figure is reported as approximate. Runs made with the current script carry
an exact interval and are left untouched.

    python scripts/backfill_ci.py b77_full_0.5b.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevmini.calibration import bootstrap_ci


def reconstruct(bins: list[dict]) -> tuple[list[float], list[bool]]:
    """Rebuild (confidence, correct) pairs from the saved reliability bins."""
    conf: list[float] = []
    correct: list[bool] = []
    for b in bins:
        n = int(b["n"])
        if n == 0:
            continue
        n_right = round(n * float(b["accuracy"]))
        conf.extend([float(b["confidence"])] * n)
        correct.extend([True] * n_right + [False] * (n - n_right))
    return conf, correct


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: backfill_ci.py <results.json> [...]")

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        d = json.loads(path.read_text(encoding="utf-8"))

        if "ece_ci" in d:
            print(f"{path.name}: already has an exact interval, skipping")
            continue
        if "bins" not in d:
            print(f"{path.name}: no bins recorded, cannot reconstruct")
            continue

        conf, correct = reconstruct(d["bins"])
        if len(conf) != d.get("n", len(conf)):
            print(f"{path.name}: bins sum to {len(conf)} but n={d.get('n')}; "
                  f"proceeding with the bin total")

        boot = bootstrap_ci(conf, correct, n_boot=2000, n_bins=5)

        # Sanity: the reconstruction should land on the ECE already recorded.
        # A large gap means the bins do not describe the run they came from.
        drift = abs(boot["ece"] - d["ece"])
        flag = "ok" if drift < 0.02 else "CHECK"
        print(f"{path.name}: reconstructed n={len(conf)}  "
              f"ECE {boot['ece']:.4f} vs recorded {d['ece']:.4f} "
              f"(drift {drift:.4f}, {flag})")

        d["accuracy_ci"] = list(boot["accuracy_ci"])
        d["ece_ci"] = list(boot["ece_ci"])
        d["ci_method"] = "bootstrap over bin reconstruction (approximate)"
        d["ci_note"] = (
            "Rebuilt from reliability bins, not per-item records: every item "
            "in a bin is placed at the bin's mean confidence, so the interval "
            "is slightly narrower than one computed from raw pairs."
        )
        d["n_boot"] = 2000

        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"  accuracy {d['accuracy']:.4f} "
              f"95% CI [{d['accuracy_ci'][0]:.4f}, {d['accuracy_ci'][1]:.4f}]")
        print(f"  ECE      {d['ece']:.4f} "
              f"95% CI [{d['ece_ci'][0]:.4f}, {d['ece_ci'][1]:.4f}]")


if __name__ == "__main__":
    main()
