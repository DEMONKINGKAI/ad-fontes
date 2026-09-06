"""Layer 2: NLI entailment check.

``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` cross-encoder (brief §4), loaded
once, run once per answer on (premise = cited chunk text, hypothesis = claim).
Per the Phase 0 decision NLI runs only on the final claims pass.

Domain note: this model was trained on MNLI/FEVER/ANLI and transfers well to
*proposition-shaped* text (as in fons-iuris's legal corpus). Portfolio prose is
more descriptive and the base generator wraps facts in framing
("Kai's design philosophy emphasises …", "Kai has expertise in …"). DeBERTa-base
rates a framed claim *neutral* — sometimes *contradiction* — even when the
factual core is verbatim in the premise. So ``score_claim``:

  * splits the claim into sentences (base-model claims are often run-ons);
  * for each sentence also tries a **frame-stripped** variant;
  * entailment = the weakest sentence's best (sentence × variant × window) score
    — every part must be supported;
  * contradiction = the strongest sentence's best-contradiction score — any
    contradicted part taints the claim.

``score_prose`` applies the same framed/stripped × window idea to a *batch* of
narration sentences in one forward pass (``verify_prose`` calls it for every
prose sentence no supported claim covers). All premise windows are hard-capped
in length so one markdown-table chunk can't force the whole batch to pad to the
model's max sequence length.

The label order is read from ``model.config.id2label``, not assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9])")
_MAX_WINDOW_CHARS = 600
_MAX_WINDOWS_PER_PREMISE = 5
_MAX_PAIRS_PER_CLAIM = 12
# Prose verification scores every narration sentence, so it batches all
# (sentence-variant, window) pairs into one forward pass and caps the shared
# window pool harder than the per-claim path.
_MAX_WINDOWS_PROSE = 8
_NLI_SUB_BATCH = 16

# Leading framing to strip so the factual core is what NLI sees.
_FRAME = re.compile(
    r"^(?:kai(?:'s)?|archit(?:\s+sharma)?|he|the\s+\w+)\b[^,.:;]*?\b"
    r"(?:is|are|was|were|has|have|had|emphasi[sz]es?|includes?|uses?|used|"
    r"centers?\s+on|focuses?\s+on|demonstrates?|shows?|reflects?|features?)\s+"
    r"(?:that\s+|a\s+|an\s+|the\s+)?",
    re.I,
)
_MARKDOWN = re.compile(r"\*{1,3}|`|_{2,}")


def _strip_frame(claim: str) -> str:
    m = _FRAME.match(claim)
    return claim[m.end() :].strip() if m and len(claim) - m.end() > 12 else claim


def _clean(text: str) -> str:
    return _MARKDOWN.sub("", text)


@dataclass(slots=True, frozen=True)
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float

    @classmethod
    def zero(cls) -> NLIScore:
        return cls(0.0, 1.0, 0.0)


_SEP_SPLIT = re.compile(r"(?<=[|\n;,])\s+|\s+")


def _hard_split(text: str, limit: int) -> list[str]:
    """Break a string with no sentence punctuation (markdown tables, comma lists)
    into <= ``limit``-char pieces so a window is never a 500-token blob that
    forces the whole NLI batch to pad to max length."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    buf = ""
    for tok in _SEP_SPLIT.split(text):
        if buf and len(buf) + len(tok) + 1 > limit:
            parts.append(buf.strip())
            buf = tok
        else:
            buf = f"{buf} {tok}".strip()
    if buf:
        parts.append(buf.strip())
    return [p for p in parts if p]


def _windows(premise: str) -> list[str]:
    premise = _clean(premise.strip())
    if len(premise) <= _MAX_WINDOW_CHARS:
        return [premise]
    sents = [
        piece for s in _SENT_SPLIT.split(premise) for piece in _hard_split(s, _MAX_WINDOW_CHARS)
    ]
    out: list[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + len(s) > _MAX_WINDOW_CHARS:
            out.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur.strip())
    for a, b in pairwise(sents):
        pair = f"{a} {b}".strip()
        if len(pair) <= _MAX_WINDOW_CHARS:
            out.append(pair)
    return (out or [premise])[:_MAX_WINDOWS_PER_PREMISE]


class NLIVerifier:
    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self._model.eval()
        self._device = device
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._idx = {
            "entailment": next(i for i, v in id2label.items() if "entail" in v),
            "neutral": next(i for i, v in id2label.items() if "neutral" in v),
            "contradiction": next(i for i, v in id2label.items() if "contradict" in v),
        }

    def _score_batch(self, premises: list[str], hypotheses: list[str]) -> list[NLIScore]:
        """NLI over parallel (premise, hypothesis) lists, in sub-batches."""
        torch = self._torch
        out: list[NLIScore] = []
        with torch.no_grad():
            for i in range(0, len(premises), _NLI_SUB_BATCH):
                pj = premises[i : i + _NLI_SUB_BATCH]
                hj = hypotheses[i : i + _NLI_SUB_BATCH]
                enc = self._tok(
                    pj,
                    hj,
                    return_tensors="pt",
                    truncation=True,
                    # windows are hard-capped at _MAX_WINDOW_CHARS (~180 tokens);
                    # 384 covers premise + hypothesis with headroom and keeps the
                    # padded batch small (the CPU forward cost is ~linear in it).
                    max_length=384,
                    padding=True,
                ).to(self._device)
                probs = torch.softmax(self._model(**enc).logits, dim=-1).cpu().tolist()
                out.extend(
                    NLIScore(
                        entailment=p[self._idx["entailment"]],
                        neutral=p[self._idx["neutral"]],
                        contradiction=p[self._idx["contradiction"]],
                    )
                    for p in probs
                )
        return out

    def _score_pairs(self, premises: list[str], hypothesis: str) -> list[NLIScore]:
        return self._score_batch(premises, [hypothesis] * len(premises))

    def score(self, premise: str, hypothesis: str) -> NLIScore:
        return self._best(self._score_pairs(_windows(premise), _clean(hypothesis)))

    def score_claim(self, claim: str, premises: list[str]) -> NLIScore:
        windows: list[str] = []
        for p in premises:
            windows.extend(_windows(p))
        windows = windows[:_MAX_PAIRS_PER_CLAIM]
        if not windows:
            return NLIScore.zero()

        sentences = [s.strip() for s in _SENT_SPLIT.split(claim.strip())]
        parts = [s for s in sentences if len(s) > 8] or [claim.strip()]

        entailments: list[float] = []
        contradictions: list[float] = []
        for part in parts:
            full = self._score_pairs(windows, _clean(part))
            stripped_v = _clean(_strip_frame(part))
            stripped = self._score_pairs(windows, stripped_v) if stripped_v != _clean(part) else []
            # entailment: best over the framed + stripped variants (framing hurts NLI)
            entailments.append(max((s.entailment for s in full + stripped), default=0.0))
            # contradiction: from the *framed* sentence only — stripping the
            # subject could mask a genuinely contradicted assertion
            contradictions.append(max((s.contradiction for s in full), default=0.0))

        e = min(entailments) if entailments else 0.0
        c = max(contradictions) if contradictions else 0.0
        return NLIScore(entailment=e, neutral=max(0.0, 1.0 - e - c), contradiction=c)

    def score_prose(self, sentences: list[str], premises: list[str]) -> list[NLIScore]:
        """One score per narration sentence against a shared premise set.

        Called by ``verify_prose`` for every prose sentence that no *supported*
        claim covers, in one batched forward pass: entailment = best over
        {sentence, frame-stripped} × windows; contradiction from the framed form
        only (as in ``score_claim``). ``premises`` should be most-relevant-first
        — the shared window pool is capped.
        """
        windows: list[str] = []
        for p in premises:
            windows.extend(_windows(p))
            if len(windows) >= _MAX_WINDOWS_PROSE:
                break
        windows = windows[:_MAX_WINDOWS_PROSE]
        if not windows or not sentences:
            return [NLIScore.zero() for _ in sentences]

        prem: list[str] = []
        hyp: list[str] = []
        owner: list[int] = []
        framed: list[bool] = []
        for i, sent in enumerate(sentences):
            clean = _clean(sent)
            variants = [(clean, True)]
            stripped = _clean(_strip_frame(sent))
            if stripped != clean:
                variants.append((stripped, False))
            for text, is_framed in variants:
                for w in windows:
                    prem.append(w)
                    hyp.append(text)
                    owner.append(i)
                    framed.append(is_framed)

        scores = self._score_batch(prem, hyp)
        ent = [0.0] * len(sentences)
        con = [0.0] * len(sentences)
        for o, is_framed, sc in zip(owner, framed, scores, strict=True):
            ent[o] = max(ent[o], sc.entailment)
            if is_framed:
                con[o] = max(con[o], sc.contradiction)
        return [NLIScore(e, max(0.0, 1.0 - e - c), c) for e, c in zip(ent, con, strict=True)]

    @staticmethod
    def _best(scores: list[NLIScore]) -> NLIScore:
        return max(scores, key=lambda s: (s.entailment, -s.contradiction), default=NLIScore.zero())
