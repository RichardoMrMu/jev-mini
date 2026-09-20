# jev-mini

**Put Jev's three headline claims on trial, on your own GPU.**

[中文文档](README.zh-CN.md)

```bash
python quickstart.py
```

One command. Downloads a 0.5B model (~1 GB, via ModelScope, so it works from
mainland China without a VPN) and then tests each of the three selling points of
a "System One Model" against a model you can actually inspect.

Runs in about two minutes on an **RTX 3060 Laptop (6 GB, browser still open)**.

---

## Why this repo exists

When Jev launched, the community objections were specific: parallel structured
output is not new, "0% hallucinations" redefines the word, and the one claim
that would genuinely matter for production — calibrated confidence — **has no
published evidence and no independent test**.

So measure it yourself. This is not a Jev clone. It is a **yardstick you can
run, read and modify**: the same local model answering the same questions two
ways — constrained scoring vs. autoregressive JSON — plus the metric everyone
discusses and nobody quantifies.

**The short version: the first two claims are real but oversold, and the third
is much harder to deliver than the marketing suggests.**

---

## The mechanism, in one sentence

> Don't let the model *write* an answer. Let it *score* the answers you allow.

A generative model answers a multiple-choice question by emitting tokens until
it has spelled out a word. Here not a single token is generated: each candidate
option is appended to the prompt, the model's log-probability for that
continuation is read off, the highest one is the decision, and the normalised
distribution is the confidence.

Three consequences follow, mapping onto exactly what TypeSafe advertises:

| The claim | How it works here | Verdict from measurement |
|---|---|---|
| Typed probabilistic decisions | Output is an index into a list | ✅ Holds |
| Zero hallucinations | Structurally impossible to go out of range | ⚠️ Holds, but it is only a *type* guarantee |
| Calibrated confidence (RLCD) | Probabilities *are* the return value, so measurable | ❌ Poor out of the box — and **a bigger model was worse** |

---

## Measured results

All numbers below come from actual runs on this machine.

Hardware: RTX 3060 Laptop 6 GB · torch 2.9.1+cu126 · Python 3.14 · Windows
Data: **120 support tickets + 60 reviews**, hand-labelled, **annotated before
any model was run**, stratified into easy / medium / hard. Ambiguous items are
deliberately included, because **calibration only means anything where a model
ought to hesitate**.

Raw output: [`results_0.5b.json`](results_0.5b.json) · [`results_1.5b.json`](results_1.5b.json)

### 1. Constrained scoring vs. autoregressive JSON

Same weights, same questions, only the output mechanism differs.

| | constrained | JSON generate |
|---|---|---|
| median latency (0.5B) | **126 ms** | 6389 ms |
| median latency (1.5B) | **127 ms** | 3531 ms |
| tokens generated | **0** | 96 / 58 |
| type errors (0.5B) | **0 / 120** | 3 / 120 |
| type errors (1.5B) | **0 / 120** | 4 / 120 |

**50.8x faster on 0.5B, 27.8x on 1.5B.** The speedup comes entirely from
removing the decode step, not from the model being smarter.

Worth noting: on 1.5B the constrained path is also *more accurate*
(0.550 vs 0.508). Same weights, same questions — the generative path loses
accuracy because tokens go into reproducing a format rather than into judging.

### 2. Parallel fields: does K questions really cost ~1 pass?

| fields | 0.5B | 1.5B | options scored | forward passes |
|---|---|---|---|---|
| 1 | 128 ms | 135 ms | 5 | 2 |
| 8 | **126 ms** | **138 ms** | 26 | 2 |

**Eight fields cost 0.98x / 1.02x the latency of one.** Within measurement
noise, asking eight questions is free. This claim holds up: the shared prefix
is computed once, its KV cache is reused, and every candidate is scored in one
batch.

### 3. Calibration — the claim with no published evidence

| Task | Model | Accuracy | Mean conf. | ECE | Verdict |
|---|---|---|---|---|---|
| Ticket category | 0.5B | 0.517 | 0.637 | 0.120 | overconfident |
| Urgency | 0.5B | 0.608 | 0.594 | **0.084** | roughly honest |
| Sentiment | 0.5B | 0.350 | 0.771 | **0.421** | badly overconfident |
| Ticket category | 1.5B | 0.550 | 0.815 | **0.265** | badly overconfident |
| Urgency | 1.5B | 0.725 | 0.927 | 0.208 | badly overconfident |
| Sentiment | 1.5B | 0.433 | 0.757 | 0.324 | badly overconfident |

**The single most telling row:** on urgency, 1.5B places **109 of 120 items
above 0.8 confidence**, where its actual accuracy is 0.734. It has essentially
one setting: certain.

**The counterintuitive finding: the bigger model got more accurate and less
calibrated.** On urgency, 0.5B is very nearly honest (ECE 0.084, marginally
*under*confident at −0.014) while 1.5B is overconfident by +0.202. Every single
1.5B task is overconfident; 0.5B is not.

Operationally, at a 0.50 auto-execute threshold:

| | auto-executed | errors let through |
|---|---|---|
| 0.5B | 78.3% | 32.5% |
| 1.5B | 96.7% | **43.3%** |

> "A bigger model gives more trustworthy confidence" — not supported here.

Also visible in the difficulty breakdown: confidence barely tracks difficulty
at all. On 0.5B ticket category, easy items get 0.640 confidence and hard items
get 0.639 — while accuracy drops from 0.475 to 0.450. The model does not know
which questions are hard.

### 4. Temperature scaling: how much does one fitted scalar recover?

| Model | 5-fold T | baseline ECE | after (CV) | after (single split) |
|---|---|---|---|---|
| 0.5B | 1.860 ± 0.190 | 0.128 | 0.085 (−34%) | **0.030 (−76%)** |
| 1.5B | 2.151 ± 0.161 | 0.311 | **0.089 (−71%)** | 0.203 (−35%) |

**Both models improve substantially, and accuracy is mathematically unchanged**
— temperature cannot move an argmax. A single scalar, fitted on a few dozen
labelled examples, cuts ECE by 71% on the worse-calibrated model.

That is the number to hold RLCD against. If a claimed calibration advantage can
be matched by fitting one parameter on held-out data, it does not warrant a new
training paradigm. Settling that properly needs TypeSafe to publish their data.

> **An earlier version of this repo, with only 18 tickets, found that
> temperature scaling made calibration *worse*.** That result was noise: at
> n=18 a single flipped item moves a 5-bin ECE by several points. It flipped to
> a clear, consistent improvement at n=120. This is left documented here
> because it is a useful caution about small-n calibration claims — including
> the ones in this README.

### 5. Routing: what the confidences actually buy you (0.5B, ticket category)

| Threshold | Auto-executed | Accuracy when auto | Errors let through |
|---|---|---|---|
| ≥0.50 | 78.3% | 0.585 | 32.5% |
| ≥0.70 | 33.3% | 0.675 | 10.8% |
| ≥0.80 | 20.0% | 0.833 | 3.3% |
| ≥0.95 | 3.3% | **1.000** | **0%** |

This table is what calibration is *for*: it is the contract between the model's
stated confidence and your automation policy.

---

## A side finding: label wording moves the probabilities

Same question, only the yes/no surface form changed. Separation between an
extreme positive and an extreme negative case nearly halves:

| Labels | Separation |
|---|---|
| `yes` / `no` | **+0.594** |
| `Yes` / `No` | +0.566 |
| `true` / `false` | +0.515 |
| `urgent` / `not urgent` | +0.316 |

If you take this approach to production, **label wording is a hyperparameter**,
not an afterthought.

---

## Usage

```python
from jevmini import JevMini, Schema, Choice, Noul, Score

schema = Schema(fields=[
    Choice(name="category", options=["billing", "technical", "account"],
           question="Which department should handle this?"),
    Noul(name="urgent",   question="Is this urgent?"),
    Score(name="severity", low=1, high=5, question="How severe?"),
])

d = engine.decide("I was charged twice for order #4471.", schema)

d.value("category")       # 'billing' -- guaranteed one of the three options
d.confidence("category")  # 0.718
d["category"]["distribution"]
# {'billing': 0.718, 'account': 0.28, 'technical': 0.002, ...}

if d.confidence("category") > 0.8:
    auto_route(d.value("category"))
else:
    escalate_to_human()
```

`Score` returns an **expected value** as well as the argmax: a review genuinely
split between 2 and 4 stars should read as 3, not as whichever bucket won by a
hair.

---

## Install

```bash
git clone https://github.com/RichardoMrMu/jev-mini
cd jev-mini
python -m venv .venv && .venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers modelscope

python quickstart.py
```

**From mainland China**: models come from ModelScope, so no HuggingFace access
is needed. Add `-i https://pypi.tuna.tsinghua.edu.cn/simple` to pip if useful.

**Full report**:

```bash
python scripts/benchmark.py --model <local model path> --out results.json
```

### Small GPU, or tight on RAM

This repo was developed in exactly that situation (6 GB card with ~2 GB already
taken by the desktop, 16 GB system RAM). `quickstart.py` sets
`TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP=1` by default, which stops
transformers from reserving a block the size of the whole model up front — that
reservation fails on a shared card even when the weights themselves would fit.

Measured peak VRAM: **0.5B → 1569 MB, 1.5B → 3582 MB**.

If you hit `OSError 1455` (pagefile too small), that is system commit memory,
not VRAM. Shut down WSL (`wsl --shutdown`) or close other large processes.

---

## What this repo does **not** prove

Stating this plainly, because leaving it out would be its own form of hype:

- **This is not Jev.** Its weights, architecture and parameter count are not
  public. This is a general-purpose small model plus constrained scoring. The
  numbers describe *this method on this machine*.
- **It does not refute TypeSafe's figures.** Their 193x / 444x are the maximum
  gaps on particular workflows; 27.8–50.8x here is a different task set, a
  different model and a different card.
- **The dataset is 120 + 60 items.** Larger than the 18 + 12 this started with —
  which was small enough to produce a spurious result — but still modest. ECE on
  a few dozen items per bin carries real uncertainty, which is why fold variance
  is reported rather than a single flattering number.
- **RLCD is not implemented.** Its training procedure and reward function are
  unpublished and cannot be reproduced. What is here is the **apparatus for
  measuring** it.

What it does establish: the type safety and parallel speedup from constrained
scoring are real and easy to obtain; calibration — the hard part — is poor on a
model not specifically trained for it, **does not improve by scaling the model
up**, and responds well to a single fitted temperature.

---

## Layout

```
jevmini/
  core.py          engine: constrained scoring, KV-cache reuse, Choice/Score/Noul
  calibration.py   ECE / MCE / Brier / NLL, temperature scaling (with CV), routing
  datasets.py      hand-labelled data, stratified by difficulty
scripts/
  benchmark.py     the full five-section report
quickstart.py      one-command demo
```

The code is commented, particularly around two traps worth knowing:
**why only the label tokens are scored** (including the question stem flattens
the distribution — a real bug hit during development: billing sat at 0.298
before the fix, 0.718 after), and **why length normalisation is required**.

## License

MIT
