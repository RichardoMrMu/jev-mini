"""
banking77: a public benchmark, for cross-validating the hand-built results.

The hand-labelled set in `datasets.py` has an obvious weakness: the author who
wrote the labels also wrote the analysis. banking77 removes that objection --
it is a standard intent-classification benchmark (Casanueva et al., 2020), the
labels are not ours, and anyone can re-run this against the same data.

It is also a harder and more revealing test of calibration, for two reasons:

  * 77 classes instead of 5. Random guessing scores 1.3%, so a confidently
    wrong answer is much more clearly wrong, and there is far more room for
    the probability mass to be misplaced.
  * Perfectly balanced -- exactly 40 examples per intent in the test split --
    so accuracy cannot be inflated by a skewed prior, and per-class behaviour
    is directly comparable.

Fetched from ModelScope rather than HuggingFace, so it works from mainland
China without a VPN. ~374 KB zipped, downloaded once and cached.

    @inproceedings{Casanueva2020,
        author    = {Casanueva, I. and Temcinas, T. and Gerz, D. and
                     Henderson, M. and Vulic, I.},
        title     = {Efficient Intent Detection with Dual Sentence Encoders},
        booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
        year      = {2020},
        url       = {https://arxiv.org/abs/2003.04807}
    }

Licensed CC-BY-4.0.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

OSS_TREE_API = "https://www.modelscope.cn/api/v1/datasets/modelscope/banking77/oss/tree"

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".data"


def _download(cache_dir: Path) -> Path:
    """Fetch and unpack banking.zip, unless it is already unpacked."""
    extracted = cache_dir / "banking" / "banking" / "test.json"
    if extracted.exists():
        return extracted

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "banking.zip"

    if not zip_path.exists():
        # The tree endpoint hands back a short-lived signed OSS URL, so it has
        # to be resolved at download time rather than hard-coded.
        with urllib.request.urlopen(OSS_TREE_API, timeout=60) as r:
            meta = json.loads(r.read().decode("utf-8"))
        url = meta["Data"][0]["Url"]
        print(f"  downloading banking77 ({meta['Data'][0]['Size']:,} bytes) ...", flush=True)
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache_dir / "banking")

    if not extracted.exists():
        raise FileNotFoundError(f"expected {extracted} after extracting {zip_path}")
    return extracted


def load_banking77(
    split: str = "test",
    limit: int | None = None,
    per_class: int | None = None,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> tuple[list[dict], list[str]]:
    """Return (items, intents).

    Each item is `{"text": ..., "intent": ...}`; `intents` is the sorted label
    space, which doubles as the option list for a Choice field.

    `per_class` samples a fixed number of examples from every intent, which is
    the right way to subsample here: taking the first N items would silently
    drop whole classes, and calibration measured over a truncated label space
    is not comparable to the full one.
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = _download(cache)
    if split != "test":
        path = path.parent / f"{split}.json"

    raw = json.loads(path.read_text(encoding="utf-8"))

    items = []
    for _, case in raw.items():
        for turn in case["turns"]:
            intent = next(iter(turn["label"]["DEFAULT_DOMAIN"]))
            items.append({"text": turn["text"], "intent": intent})

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

    if limit:
        items = items[:limit]

    return items, intents


def humanise(intent: str) -> str:
    """'card_arrival' -> 'card arrival'.

    Underscored identifiers are not what the model saw during pre-training, and
    scoring them directly measures tokenizer luck as much as understanding.
    The label wording experiment in the README showed this matters: separation
    between clear-cut cases changed by nearly half depending on surface form.
    """
    return intent.replace("_", " ")
