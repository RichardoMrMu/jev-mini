"""
Check every figure the READMEs quote against the file it came from.

Three stale numbers surfaced in this repo one after another -- a confidence of
0.718 that no longer reproduced, a type-error count of 3 that was really 10,
and a 50.8x speedup that was really 39.0x. Each was found by accident. They
share one cause: a result file gets regenerated, or a code path changes, and
the prose keeps the old value. Nothing fails, so nothing announces it.

That is a poor failure mode anywhere, and a disqualifying one here: this repo
exists to check other people's unverified numbers. So the check runs on
demand instead of relying on someone noticing.

Each entry pins a figure in the README to the JSON field that must produce it.
Reads only; no GPU, no model, about a second.

    python scripts/audit_figures.py

Exit code 0 when everything agrees, 1 otherwise, so it can gate a commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS = (
    "results_0.5b", "results_1.5b",
    "b77_full_0.5b", "b77_full_1.5b",
    "clinc_0.5b", "chunk_sensitivity",
)


def load() -> dict:
    out = {}
    for name in RESULTS:
        p = ROOT / f"{name}.json"
        if not p.exists():
            sys.exit(f"missing {p.name}; run from a full checkout")
        out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def present(val: str, text: str) -> bool:
    """The Chinese README writes speedups as 快 39.0 倍, not 39.0x."""
    if val in text:
        return True
    if val.endswith("x"):
        return val[:-1] in text
    return False


def main() -> None:
    J = load()
    EN = (ROOT / "README.md").read_text(encoding="utf-8")
    ZH = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    s0, s1 = J["results_0.5b"]["speed"], J["results_1.5b"]["speed"]
    b5, b15 = J["b77_full_0.5b"], J["b77_full_1.5b"]
    c5, cs = J["clinc_0.5b"], J["chunk_sensitivity"]

    checks = [
        ("s1 0.5B constrained ms", f"{s0['constrained_median_ms']}"),
        ("s1 0.5B generate ms",    f"{s0['generate_median_ms']}"),
        ("s1 1.5B constrained ms", f"{s1['constrained_median_ms']}"),
        ("s1 1.5B generate ms",    f"{s1['generate_median_ms']}"),
        ("s1 0.5B type errors",    f"{s0['generate_type_errors']} / 120"),
        ("s1 1.5B type errors",    f"{s1['generate_type_errors']} / 120"),
        ("s1 0.5B speedup",        f"{s0['speedup']}x"),
        ("s1 1.5B speedup",        f"{s1['speedup']}x"),
        ("b77 0.5B accuracy", f"{b5['accuracy']:.3f}"),
        ("b77 1.5B accuracy", f"{b15['accuracy']:.3f}"),
        ("b77 0.5B ECE",      f"{b5['ece']:.4f}"),
        ("b77 1.5B ECE",      f"{b15['ece']:.4f}"),
        ("b77 0.5B CI low",   f"{b5['ece_ci'][0]:.4f}"),
        ("b77 0.5B CI high",  f"{b5['ece_ci'][1]:.4f}"),
        ("b77 1.5B CI low",   f"{b15['ece_ci'][0]:.4f}"),
        ("b77 1.5B CI high",  f"{b15['ece_ci'][1]:.4f}"),
        ("clinc accuracy",    f"{c5['accuracy']:.3f}"),
        ("clinc ECE",         f"{c5['ece']:.4f}"),
        ("clinc fitted T",    f"{c5['temperature']['T_cv']:.3f}"),
        ("clinc ECE after T", f"{c5['temperature']['ece_after']:.4f}"),
        ("chunk 8 ECE",  f"{cs['chunks']['8']['ece']:.6f}"),
        ("chunk 20 ECE", f"{cs['chunks']['20']['ece']:.6f}"),
        ("chunk 77 ECE", f"{cs['chunks']['77']['ece']:.6f}"),
        ("chunk accuracy", f"{cs['chunks']['8']['accuracy']:.4f}"),
    ]

    bad = 0
    for name, val in checks:
        ie, iz = present(val, EN), present(val, ZH)
        if not (ie and iz):
            print(f"  DRIFT  {name:22} expected {val!r}   en={ie} zh={iz}")
            bad += 1
    print(f"{len(checks) - bad}/{len(checks)} figures match their source file")

    # Values that were corrected and must not creep back via a stale edit.
    for r in ("0.718", "50.8", "6389", "3531", "0.508"):
        if r in EN or r in ZH:
            print(f"  RETIRED FIGURE PRESENT: {r} (en={r in EN}, zh={r in ZH})")
            bad += 1

    # The qualitative claims, checked against the data rather than the prose.
    claims = [
        (b5["ece_ci"][1] < b15["ece_ci"][0],
         "banking77: the two ECE intervals are disjoint"),
        (b5["overconfidence"] < 0 and c5["overconfidence"] < 0,
         "77 and 150 classes are both under-confident"),
        (c5["temperature"]["ece_after"] > c5["temperature"]["ece_before"],
         "temperature scaling makes CLINC150 worse"),
        (cs["ece_spread"] < (b15["ece"] - b5["ece"]) / 5,
         "chunk noise stays well under the headline gap"),
        (cs["predictions_changed"] == 0,
         "chunking changes no predictions"),
        (s1["generate_type_errors"] > s0["generate_type_errors"],
         "the larger model breaks the output contract more often"),
    ]
    for ok, msg in claims:
        print(("  OK   " if ok else "  FAIL ") + msg)
        if not ok:
            bad += 1

    print()
    if bad:
        print(f"RESULT: {bad} problem(s) -- fix the README or re-measure")
        sys.exit(1)
    print("RESULT: every published figure matches its source")


if __name__ == "__main__":
    main()
