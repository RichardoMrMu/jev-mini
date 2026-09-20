"""
jev-mini core: a minimal, local reimplementation of the "System One Model" idea.

The mechanism in one sentence:
    Do NOT let the model write an answer. Let it *score* the answers you allow.

A generative LLM answers a multiple-choice question by emitting tokens until it
has spelled out a word. We never let it emit anything. Instead we take each
allowed option, append it to the prompt, and read off the log-probability the
model assigns to that continuation. The option with the highest probability is
the decision; the normalised distribution over options is the confidence.

Three consequences fall out of this, and they map 1:1 onto the three things
TypeSafe advertises for Jev:

1. Type safety is structural, not statistical.
   The output is an index into a Python list. There is no decoding step that
   could produce a fourth category, a stray apostrophe, or a 200-word apology.
   "0% type errors" is not a benchmark result here either -- it is arithmetic.

2. Many questions cost one forward pass, not N.
   Scoring is a pure read of the logits, so K questions about the same input
   share the same prefix. We run the prefix once, keep the KV cache, and fan out
   over every (question, option) pair in a single batched continuation.

3. The probabilities are inspectable.
   Autoregressive JSON throws the distribution away: "refund" is just a string,
   and 51% certainty and 99% certainty print identically. Here the distribution
   IS the return value, which is what makes calibration measurable at all.

Nothing in this file is specific to Qwen; it needs a causal LM and a tokenizer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Field types -- the three output shapes Jev exposes (Choice / Score / Noul)
# --------------------------------------------------------------------------


@dataclass
class Choice:
    """Pick exactly one of `options`."""

    name: str
    options: Sequence[str]
    question: str = ""

    @property
    def labels(self) -> list[str]:
        return list(self.options)

    def render_question(self) -> str:
        return self.question or f"What is the {self.name.replace('_', ' ')}?"

    def decode(self, probs: torch.Tensor) -> dict[str, Any]:
        idx = int(probs.argmax())
        return {
            "type": "choice",
            "value": self.labels[idx],
            "confidence": float(probs[idx]),
            "distribution": {l: float(p) for l, p in zip(self.labels, probs)},
        }


@dataclass
class Noul:
    """A yes/no judgement. (Jev's name for a probabilistic boolean.)"""

    name: str
    question: str = ""
    yes_label: str = "yes"
    no_label: str = "no"

    @property
    def labels(self) -> list[str]:
        return [self.no_label, self.yes_label]

    def render_question(self) -> str:
        return self.question or f"Is it {self.name.replace('_', ' ')}?"

    def decode(self, probs: torch.Tensor) -> dict[str, Any]:
        p_yes = float(probs[1])
        return {
            "type": "noul",
            "value": p_yes >= 0.5,
            "confidence": p_yes if p_yes >= 0.5 else 1.0 - p_yes,
            "p_true": p_yes,
        }


@dataclass
class Score:
    """An ordinal rating. Returns both the modal bucket and the expected value.

    The expected value is the useful part: averaging over the whole distribution
    keeps the information that argmax throws away. A review that is genuinely
    split between 2 and 4 stars should read as 3, not as whichever bucket won by
    a hair.
    """

    name: str
    low: int = 1
    high: int = 5
    question: str = ""

    @property
    def labels(self) -> list[str]:
        return [str(i) for i in range(self.low, self.high + 1)]

    def render_question(self) -> str:
        return self.question or (
            f"On a scale of {self.low} to {self.high}, "
            f"what is the {self.name.replace('_', ' ')}?"
        )

    def decode(self, probs: torch.Tensor) -> dict[str, Any]:
        values = torch.arange(self.low, self.high + 1, dtype=probs.dtype)
        expected = float((probs * values).sum())
        idx = int(probs.argmax())
        return {
            "type": "score",
            "value": int(self.labels[idx]),
            "expected": round(expected, 3),
            "confidence": float(probs[idx]),
            "distribution": {l: float(p) for l, p in zip(self.labels, probs)},
        }


Field = Choice | Noul | Score


@dataclass
class Schema:
    """A bundle of fields evaluated together against one input."""

    fields: list[Field]
    instruction: str = "You are a precise classifier."

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate field names: {sorted(dupes)}")
        if not self.fields:
            raise ValueError("schema needs at least one field")


@dataclass
class Decision:
    """What a single `decide()` call returns."""

    fields: dict[str, Any]
    latency_ms: float
    forward_passes: int
    prefix_tokens: int
    scored_continuations: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def value(self, key: str) -> Any:
        return self.fields[key]["value"]

    def confidence(self, key: str) -> float:
        return self.fields[key]["confidence"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": self.fields,
            "latency_ms": round(self.latency_ms, 2),
            "forward_passes": self.forward_passes,
            "prefix_tokens": self.prefix_tokens,
            "scored_continuations": self.scored_continuations,
            **self.meta,
        }


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class JevMini:
    """Constrained-scoring decision engine over a local causal LM."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str | None = None,
        temperature: float = 1.0,
        length_norm: bool = True,
        max_chunk: int | None = None,
    ) -> None:
        self.model = model
        self.tok = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Temperature here is a *calibration* knob applied to the option-level
        # log-likelihoods, not a sampling temperature. T>1 softens overconfident
        # distributions; T<1 sharpens them. Fit one with
        # jevmini.calibration.fit_temperature_cv on held-out labelled data.
        self.temperature = temperature
        # Options differ in token length, and longer strings accumulate more
        # negative log-prob purely by being longer. Without normalisation the
        # model reliably prefers "spam" over "promotional" for reasons that have
        # nothing to do with the email.
        self.length_norm = length_norm
        # Options are scored in chunks so that a large label space cannot blow
        # up VRAM; None auto-sizes from free memory. Set an int to pin it.
        self.max_chunk = max_chunk
        self._last_chunks = 1
        self.model.eval()

    # -- prompt construction ------------------------------------------------

    def _build_prefix(self, text: str, schema: Schema) -> str:
        lines = [schema.instruction, "", "### Input", text.strip(), "", "### Task"]
        if len(schema.fields) > 1:
            lines.append("Answer each question independently.")
        return "\n".join(lines) + "\n\n"

    def _question_stem(self, f: Field) -> str:
        return f"Q: {f.render_question()}\nA:"

    # -- the scoring core ---------------------------------------------------

    @torch.no_grad()
    def decide(self, text: str, schema: Schema) -> Decision:
        """Score every (field, option) pair for `text` in one batched pass."""
        t0 = time.perf_counter()

        prefix = self._build_prefix(text, schema)
        prefix_ids = self.tok(prefix, return_tensors="pt").input_ids.to(self.device)
        n_prefix = prefix_ids.shape[1]

        # ---- 1. Run the shared prefix exactly once, and keep its KV cache.
        # This is the whole trick behind "K questions for the price of one".
        # The document may be 2000 tokens; the questions are a handful each.
        # Recomputing the document per question is what makes the naive
        # approach slow, and it is pure waste.
        prefix_out = self.model(prefix_ids, use_cache=True)
        past = prefix_out.past_key_values
        last_logits = prefix_out.logits[:, -1, :]

        # ---- 2. Flatten every (field, option) pair into one batch.
        #
        # Each continuation is "<question stem> <label>", but only the LABEL
        # tokens are scored. The stem conditions the model and is then excluded
        # from the sum -- it is byte-identical across the options of a field, so
        # including it would add the same large constant to every score while
        # dragging the length-normalised average toward the stem's own
        # likelihood. That flattens the distribution: the handful of tokens that
        # actually discriminate get averaged against dozens that do not.
        continuations: list[str] = []
        stems: list[str] = []
        owners: list[int] = []
        for fi, f in enumerate(schema.fields):
            stem = self._question_stem(f)
            for label in f.labels:
                continuations.append(f"{stem} {label}")
                stems.append(stem)
                owners.append(fi)

        scores = self._score_continuations(
            prefix_ids, past, last_logits, continuations, stems
        )

        # ---- 3. Softmax within each field, independently.
        results: dict[str, Any] = {}
        for fi, f in enumerate(schema.fields):
            mine = torch.tensor(
                [s for s, o in zip(scores, owners) if o == fi], dtype=torch.float32
            )
            probs = F.softmax(mine / self.temperature, dim=-1)
            results[f.name] = f.decode(probs)

        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        return Decision(
            fields=results,
            latency_ms=dt,
            # One pass for the shared prefix, then one per scoring chunk --
            # usually a single chunk, more when the label space is large
            # enough that the logits would not fit in VRAM at once.
            forward_passes=1 + getattr(self, "_last_chunks", 1),
            prefix_tokens=n_prefix,
            scored_continuations=len(continuations),
        )

    @torch.no_grad()
    def _score_continuations(
        self,
        prefix_ids: torch.Tensor,
        past,
        last_logits: torch.Tensor,
        continuations: list[str],
        stems: list[str],
    ) -> list[float]:
        """Log-prob of each option's own tokens, given the cached prefix + stem.

        Scored in chunks. One batch over every option is the fastest
        arrangement right up until the label space gets large, and then it is
        simply impossible: the logits tensor is
        [n_options, seq_len, vocab_size], which for banking77's 77 intents
        against a 151k vocabulary is hundreds of megabytes in fp16 and about a
        gigabyte once anything promotes it to fp32. On a 6 GB card already
        holding the weights that is an instant OOM. Chunking bounds peak
        memory at a slice of it, for a few extra forward passes.

        A note on reproducibility, because this is easy to misdiagnose:
        changing the chunk size moves the probabilities in roughly the third
        decimal place under fp16. That is GPU reduction order within a batch,
        not a corrupted cache. Three checks establish it -- any single
        configuration is bit-for-bit repeatable; the error does not grow with
        chunk index, which a polluted cache would guarantee; and in fp32 the
        same comparison falls from 5e-3 to 2e-6. Rankings and argmax are
        unaffected. Pin max_chunk if you need identical digits across runs.
        """
        B = len(continuations)

        enc = [self.tok(c, add_special_tokens=False).input_ids for c in continuations]
        # Token count of the stem alone, so we know where the label begins.
        # Measured by re-tokenising the stem rather than by string offsets,
        # since BPE merges across the space boundary.
        stem_lens = [
            len(self.tok(s, add_special_tokens=False).input_ids) for s in stems
        ]

        maxlen = max(len(e) for e in enc)
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id or 0

        cont = torch.full((B, maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((B, maxlen), dtype=torch.bool)
        for i, e in enumerate(enc):
            cont[i, : len(e)] = torch.tensor(e)
            # Score only the label tokens; the stem is context, not evidence.
            start = min(stem_lens[i], len(e) - 1)
            mask[i, start : len(e)] = True
        cont = cont.to(self.device)
        mask = mask.to(self.device)

        chunk = self._chunk_size(B, maxlen)
        self._last_chunks = (B + chunk - 1) // chunk
        totals: list[float] = []

        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            c_cont = cont[lo:hi]
            c_mask = mask[lo:hi]
            n = hi - lo

            # Each chunk needs its own broadcast view of the prefix cache: a
            # forward pass appends to whatever cache it is handed, so reusing
            # one across chunks would keep growing it.
            chunk_past = _expand_cache(_clone_cache(past), n)

            attn = torch.ones(
                (n, prefix_ids.shape[1] + maxlen),
                dtype=torch.long,
                device=self.device,
            )
            out = self.model(c_cont, past_key_values=chunk_past, attention_mask=attn)

            # The prefix's final logits predict each continuation's first
            # token; the continuation's own logits predict the rest. Stitching
            # them gives a next-token distribution aligned to every position.
            first = last_logits.expand(n, -1).unsqueeze(1)
            logits = torch.cat([first, out.logits[:, :-1, :]], dim=1)

            # Only the chosen token's log-prob is needed:
            #   log p(token) = logit(token) - logsumexp(all logits)
            # Both log_softmax and a bare logits.float() would allocate
            # another whole [n, seq, vocab] tensor -- 848 MB for one chunk of
            # banking77, which is itself enough to OOM the card the chunking
            # was meant to protect. So gather the one logit that matters, and
            # take the log-partition one position at a time: each step casts a
            # [n, vocab] slice rather than the entire cube.
            picked = logits.gather(2, c_cont.unsqueeze(-1)).squeeze(-1).float()
            lse = torch.empty(
                logits.shape[0], logits.shape[1],
                dtype=torch.float32, device=logits.device,
            )
            for pos in range(logits.shape[1]):
                lse[:, pos] = torch.logsumexp(logits[:, pos, :].float(), dim=-1)
            tok_lp = (picked - lse).masked_fill(~c_mask, 0.0)

            t = tok_lp.sum(dim=1)
            if self.length_norm:
                t = t / c_mask.sum(dim=1).clamp(min=1)
            totals.extend(t.tolist())

            del out, logits, picked, lse, tok_lp, chunk_past

        return totals

    def _chunk_size(self, n_options: int, seq_len: int) -> int:
        """How many options can be scored at once without exhausting VRAM.

        Sized from free memory against a per-option cost that was measured,
        not derived. Getting this wrong is not a performance question: too
        large and the run dies on the card the chunking exists to protect.
        """
        if self.max_chunk is not None:
            return max(1, self.max_chunk)
        if self.device != "cuda":
            return n_options

        vocab = getattr(self.model.config, "vocab_size", 32000)
        free, _ = torch.cuda.mem_get_info()

        # Budget a third of what is free. Free memory is sampled before the
        # forward pass, which then wants activations and workspace of its own,
        # and on a shared card the desktop can claim more at any moment.
        # Erring low costs an extra forward pass; erring high costs an OOM.
        budget = free * 0.33

        # Bytes per option, calibrated against measurement rather than theory.
        # The fp16 logits cube alone is seq_len * vocab * 2, but profiling
        # decide() on banking77 showed the true peak is about 3.4x that once
        # the concatenation, the fp32 gather, the logsumexp accumulator and
        # the allocator's own slack are counted: 77 options at seq_len 19
        # peaked at 991 MB, i.e. ~12.9 MB each against a 6.6 MB cube.
        # Under-counting here is exactly what made an earlier version pick
        # "no chunking at all" on a quiet card and then OOM on a busy one.
        per_option = seq_len * vocab * 7
        fits = int(budget // max(per_option, 1))
        return max(1, min(n_options, fits))

    # -- baseline for comparison -------------------------------------------

    @torch.no_grad()
    def generate_json_baseline(
        self, text: str, schema: Schema, max_new_tokens: int = 96
    ) -> dict[str, Any]:
        """The conventional approach, for an honest side-by-side.

        Ask the same model for JSON, let it decode token by token, then parse.
        This is what the constrained path is being compared against: same
        weights, same hardware, same questions -- only the output mechanism
        differs. It is also where type errors come from.
        """
        import json
        import re

        spec_lines = []
        for f in schema.fields:
            if isinstance(f, Choice):
                spec_lines.append(f'  "{f.name}": one of {list(f.options)}')
            elif isinstance(f, Noul):
                spec_lines.append(f'  "{f.name}": true or false')
            else:
                spec_lines.append(f'  "{f.name}": integer {f.low}-{f.high}')

        prompt = (
            f"{schema.instruction}\n\n### Input\n{text.strip()}\n\n"
            "### Task\nReturn ONLY a JSON object with these keys:\n{\n"
            + ",\n".join(spec_lines)
            + "\n}\n\nJSON:\n"
        )
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        t0 = time.perf_counter()
        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        raw = self.tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        n_new = int(out.shape[1] - ids.shape[1])

        parsed, type_error, reason = None, False, None
        m = re.search(r"\{.*?\}", raw, re.S)
        if not m:
            type_error, reason = True, "no JSON object found"
        else:
            try:
                parsed = json.loads(m.group(0))
            except Exception as e:
                type_error, reason = True, f"invalid JSON: {e}"

        # Even syntactically valid JSON can be unusable downstream: a missing
        # key, or a category the program was never written to handle.
        if parsed is not None:
            for f in schema.fields:
                if f.name not in parsed:
                    type_error, reason = True, f"missing key {f.name!r}"
                    break
                v = parsed[f.name]
                if isinstance(f, Choice) and v not in f.options:
                    type_error, reason = True, f"{f.name}={v!r} not in {list(f.options)}"
                    break
                if isinstance(f, Score):
                    try:
                        iv = int(v)
                    except Exception:
                        type_error, reason = True, f"{f.name}={v!r} not an int"
                        break
                    if not (f.low <= iv <= f.high):
                        type_error, reason = True, f"{f.name}={iv} out of range"
                        break
                if isinstance(f, Noul) and not isinstance(v, bool):
                    type_error, reason = True, f"{f.name}={v!r} not a bool"
                    break

        return {
            "parsed": parsed,
            "raw": raw,
            "latency_ms": dt,
            "new_tokens": n_new,
            "type_error": type_error,
            "error_reason": reason,
        }


def _clone_cache(past):
    """A fresh cache object over the same underlying key/value tensors.

    The model appends the continuation's keys and values to whatever cache it
    is given, so handing the same object to a second chunk would score that
    chunk against a prefix polluted by the first. Cloning the container while
    sharing the tensors keeps each chunk seeing exactly the prefix -- the
    tensors themselves are never written in place, only concatenated onto.
    """
    import copy

    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        new = copy.copy(past)
        new.key_cache = list(past.key_cache)
        new.value_cache = list(past.value_cache)
        return new

    if hasattr(past, "layers"):
        new = copy.copy(past)
        new.layers = []
        for layer in past.layers:
            nl = copy.copy(layer)
            new.layers.append(nl)
        return new

    return tuple(tuple(t for t in layer) for layer in past)


def _expand_cache(past, batch: int):
    """Repeat a batch-1 KV cache across `batch` rows.

    Works with both the legacy tuple-of-tuples layout and the newer Cache
    objects, since which one you get depends on the transformers version.
    """
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i].expand(batch, -1, -1, -1)
            past.value_cache[i] = past.value_cache[i].expand(batch, -1, -1, -1)
        return past

    if hasattr(past, "layers"):
        for layer in past.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys.expand(batch, -1, -1, -1)
                layer.values = layer.values.expand(batch, -1, -1, -1)
        return past

    return tuple(
        tuple(t.expand(batch, -1, -1, -1) for t in layer) for layer in past
    )
