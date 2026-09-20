"""
CLINC150: a second public benchmark, chosen to test one specific hypothesis.

banking77 showed that the *direction* of the calibration error flipped between
5 classes and 77 -- over-confident on the small label space, under-confident on
the large one. That is a claim about label-space size, and a claim like that
cannot be settled with two points. CLINC150 supplies a third: 150 in-scope
intents, roughly double banking77, with a random baseline of 0.67%.

If under-confidence deepens again at 150 classes, the pattern is real and the
practical lesson follows: a calibration number means nothing without the label
space it was measured on. If it does not, the banking77 result was about that
dataset rather than about label-space size, and the README has to say so.

The dataset also ships 1000 explicitly out-of-scope utterances -- queries no
intent covers. They are loaded here but kept out of the calibration numbers by
default, because "correct" is undefined for them under a forced choice. They
are the natural material for a follow-up on whether confidence detects
out-of-distribution input at all.

A warning on cost, learned the hard way: scoring 150 options per item is about
twice the work of banking77's 77, and on a 6 GB laptop GPU a 1500-item run can
saturate the machine for the better part of an hour. Start with a small
--per-class and scale up once you know the throughput on your card.

    @inproceedings{larson-etal-2019-evaluation,
        title     = {An Evaluation Dataset for Intent Classification and
                     Out-of-Scope Prediction},
        author    = {Larson, Stefan and Mahendran, Anish and others},
        booktitle = {EMNLP-IJCNLP},
        year      = {2019},
        url       = {https://www.aclweb.org/anthology/D19-1131}
    }

Source: github.com/clinc/oos-eval (CC BY 3.0). ~2.5 MB, cached after first use.
Mirrors are tried in turn because raw.githubusercontent is intermittently
unreachable from some networks -- in testing here the direct fetch timed out
and the ghproxy mirror succeeded.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

SOURCES = [
    "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
    "https://ghproxy.net/https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
    "https://cdn.jsdelivr.net/gh/clinc/oos-eval@master/data/data_full.json",
]

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".data"


def _download(cache_dir: Path) -> Path:
    path = cache_dir / "clinc150_data_full.json"
    if path.exists():
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in SOURCES:
        try:
            print(f"  downloading CLINC150 from {url.split('/')[2]} ...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            # Parse before writing: a proxy that returns an HTML error page
            # would otherwise be cached as if it were the dataset.
            json.loads(body.decode("utf-8"))
            path.write_bytes(body)
            return path
        except Exception as e:  # noqa: BLE001 - any failure means try the mirror
            last_error = e
            print(f"    failed ({type(e).__name__}), trying next source", flush=True)

    raise RuntimeError(f"could not fetch CLINC150 from any source: {last_error}")


def load_clinc150(
    split: str = "test",
    per_class: int | None = None,
    include_oos: bool = False,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Return (items, intents) with the same shape as `load_banking77`.

    `include_oos` appends the out-of-scope utterances with intent "oos". Leave
    it off for calibration: under a forced choice over 150 in-scope labels
    there is no right answer for them, so scoring them as errors would measure
    the schema's lack of an escape hatch rather than the model's confidence.
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    raw = json.loads(_download(cache).read_text(encoding="utf-8"))

    pairs = list(raw[split])
    items = [{"text": t, "intent": lab} for t, lab in pairs]
    intents = sorted({it["intent"] for it in items})

    if per_class:
        import random

        rng = random.Random(seed)
        by_intent: dict[str, list[dict]] = {}
        for it in items:
            by_intent.setdefault(it["intent"], []).append(it)
        picked = []
        for intent in intents:
            group = list(by_intent[intent])
            rng.shuffle(group)
            picked.extend(group[:per_class])
        items = picked
        rng.shuffle(items)

    if include_oos:
        oos_key = {"test": "oos_test", "val": "oos_val", "train": "oos_train"}[split]
        items.extend({"text": t, "intent": "oos"} for t, _ in raw[oos_key])

    return items, intents


def humanise(intent: str) -> str:
    """'account_blocked' -> 'account blocked'.

    Same reasoning as in banking77.py: underscored identifiers are not what the
    model saw in pre-training, and scoring them measures tokenizer luck as much
    as understanding.
    """
    return intent.replace("_", " ")
