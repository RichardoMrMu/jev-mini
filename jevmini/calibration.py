"""
Calibration metrics.

This is the part of the repo that exists because nobody has published the
evidence. Parallel structured output is a known trick, and "0% hallucinations"
is a statement about types rather than about facts. The one claim that would
genuinely matter for automation -- that a stated 90% confidence corresponds to
being right about 90% of the time -- is also the one claim that has not been
independently shown.

So measure it. A model whose confidences are meaningless cannot be used to route
work: "execute above 0.95, escalate below 0.70" is only a policy if those
numbers refer to something real.

Note what calibration is not. A model can be perfectly calibrated and useless:
if it answers every question with 25% confidence across four options and is
right a quarter of the time, its ECE is ~0 and it has told you nothing. That is
why accuracy is always reported alongside, and why the reliability diagram
matters more than any single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class CalibrationReport:
    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    mce: float
    brier: float
    nll: float
    bins: list[dict]

    @property
    def overconfidence(self) -> float:
        """Positive => claims more certainty than it earns."""
        return self.mean_confidence - self.accuracy

    def summary(self) -> str:
        arrow = "overconfident" if self.overconfidence > 0 else "underconfident"
        return (
            f"n={self.n}  acc={self.accuracy:.3f}  conf={self.mean_confidence:.3f}  "
            f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  Brier={self.brier:.4f}  "
            f"NLL={self.nll:.4f}  ({arrow} by {abs(self.overconfidence):.3f})"
        )

    def reliability_table(self) -> str:
        lines = [
            "  bin          n     conf     acc      gap",
            "  " + "-" * 44,
        ]
        for b in self.bins:
            if b["n"] == 0:
                continue
            lines.append(
                f"  [{b['lo']:.2f},{b['hi']:.2f})  {b['n']:4d}   "
                f"{b['confidence']:.3f}   {b['accuracy']:.3f}   {b['gap']:+.3f}"
            )
        return "\n".join(lines)


def compute_calibration(
    confidences: Sequence[float],
    correct: Sequence[bool],
    probs_full: Sequence[Sequence[float]] | None = None,
    true_idx: Sequence[int] | None = None,
    n_bins: int = 10,
) -> CalibrationReport:
    """Expected/Maximum Calibration Error plus Brier and NLL.

    `confidences` is the probability assigned to the *predicted* option;
    `correct` says whether that prediction was right. Pass `probs_full` and
    `true_idx` to also get proper scoring rules over the whole distribution.
    """
    import math

    assert len(confidences) == len(correct), "length mismatch"
    n = len(confidences)
    if n == 0:
        raise ValueError("no samples")

    acc = sum(1 for c in correct if c) / n
    mean_conf = sum(confidences) / n

    bins = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        # Last bin closes on the right so that confidence == 1.0 lands somewhere.
        idx = [
            j
            for j, c in enumerate(confidences)
            if (lo <= c < hi) or (i == n_bins - 1 and c == 1.0)
        ]
        if not idx:
            bins.append(
                {"lo": lo, "hi": hi, "n": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0}
            )
            continue
        b_conf = sum(confidences[j] for j in idx) / len(idx)
        b_acc = sum(1 for j in idx if correct[j]) / len(idx)
        gap = b_conf - b_acc
        ece += (len(idx) / n) * abs(gap)
        mce = max(mce, abs(gap))
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(idx),
                "confidence": b_conf,
                "accuracy": b_acc,
                "gap": gap,
            }
        )

    # Brier and NLL over the full distribution when available; otherwise the
    # binary reduction onto the predicted option.
    if probs_full is not None and true_idx is not None:
        brier = sum(
            sum((p - (1.0 if k == t else 0.0)) ** 2 for k, p in enumerate(row))
            for row, t in zip(probs_full, true_idx)
        ) / n
        nll = -sum(
            math.log(max(row[t], 1e-12)) for row, t in zip(probs_full, true_idx)
        ) / n
    else:
        brier = sum(
            (c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(confidences, correct)
        ) / n
        nll = -sum(
            math.log(max(c if ok else 1.0 - c, 1e-12))
            for c, ok in zip(confidences, correct)
        ) / n

    return CalibrationReport(
        n=n,
        accuracy=acc,
        mean_confidence=mean_conf,
        ece=ece,
        mce=mce,
        brier=brier,
        nll=nll,
        bins=bins,
    )


def fit_temperature(
    logit_rows: Sequence[Sequence[float]],
    true_idx: Sequence[int],
    lo: float = 0.05,
    hi: float = 10.0,
    steps: int = 200,
    prior_strength: float = 0.0,
) -> float:
    """Find the temperature minimising NLL -- the standard post-hoc fix.

    Temperature scaling divides the scores by a single scalar before the
    softmax. It cannot change any argmax, so accuracy is mathematically
    untouched; all it does is stretch or compress the confidences. That makes it
    the honest baseline to hold RLCD against: if a claimed calibration advantage
    can be matched by fitting one number on a handful of labelled examples, the
    advantage is not worth a new training paradigm.

    `prior_strength` shrinks the estimate toward T=1 (leave it alone). On a few
    dozen examples the unregularised optimum is largely an artefact of which
    items landed in the split, and it will happily return a T that makes
    held-out calibration worse. Set it to 0 for the textbook estimator.

    Fit this on a held-out split, never on the evaluation split.
    """
    import math

    def nll_at(T: float) -> float:
        total = 0.0
        for row, t in zip(logit_rows, true_idx):
            scaled = [s / T for s in row]
            m = max(scaled)
            lse = m + math.log(sum(math.exp(s - m) for s in scaled))
            total += lse - scaled[t]
        total /= len(logit_rows)
        if prior_strength > 0:
            total += prior_strength * (math.log(T) ** 2)
        return total

    # Coarse-to-fine scan. The objective is convex in 1/T, so this is plenty.
    best_T, best = lo, nll_at(lo)
    for i in range(1, steps + 1):
        T = lo + (hi - lo) * i / steps
        v = nll_at(T)
        if v < best:
            best, best_T = v, T

    span = (hi - lo) / steps
    left, right = max(lo, best_T - span), min(hi, best_T + span)
    for i in range(101):
        T = left + (right - left) * i / 100
        v = nll_at(T)
        if v < best:
            best, best_T = v, T

    return best_T


def fit_temperature_cv(
    logit_rows: Sequence[Sequence[float]],
    true_idx: Sequence[int],
    n_folds: int = 5,
    prior_strength: float = 0.05,
) -> tuple[float, float]:
    """Leave-fold-out temperature fitting; returns (mean_T, std_T).

    On a dataset this small, a single dev/eval split yields a temperature that
    is mostly an artefact of the split. Fitting once per fold and averaging is
    steadier -- and the spread across folds is itself the useful output: a wide
    std is the honest signal that there is not enough labelled data here to
    claim a calibration improvement at all.
    """
    import statistics

    n = len(logit_rows)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)

    Ts = []
    for f in range(n_folds):
        tr_rows = [r for i, r in enumerate(logit_rows) if i % n_folds != f]
        tr_idx = [t for i, t in enumerate(true_idx) if i % n_folds != f]
        if len(tr_rows) < 2:
            continue
        Ts.append(fit_temperature(tr_rows, tr_idx, prior_strength=prior_strength))

    if not Ts:
        return 1.0, 0.0
    return statistics.mean(Ts), (statistics.stdev(Ts) if len(Ts) > 1 else 0.0)


def routing_table(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> str:
    """What the calibration is actually *for*.

    Auto-execute above a threshold, escalate the rest. This turns abstract ECE
    into the number an operator cares about: of the work you let through
    untouched, how much of it was wrong?
    """
    n = len(confidences)
    lines = [
        "  threshold   auto%    auto-acc   escalated   err-let-through",
        "  " + "-" * 62,
    ]
    for t in thresholds:
        auto = [i for i, c in enumerate(confidences) if c >= t]
        if not auto:
            lines.append(f"  >={t:.2f}       0.0%       --          {n:4d}        --")
            continue
        a_acc = sum(1 for i in auto if correct[i]) / len(auto)
        errs = sum(1 for i in auto if not correct[i])
        lines.append(
            f"  >={t:.2f}      {100*len(auto)/n:5.1f}%     {a_acc:.3f}       "
            f"{n-len(auto):4d}        {errs:3d} ({100*errs/n:.1f}% of all)"
        )
    return "\n".join(lines)
