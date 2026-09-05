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


def _windows(premise: str) -> list[str]:
    premise = _clean(premise.strip())
    if len(premise) <= _MAX_WINDOW_CHARS:
        return [premise]
    sents = _SENT_SPLIT.split(premise)
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

    def _score_pairs(self, premises: list[str], hypothesis: str) -> list[NLIScore]:
        torch = self._torch
        with torch.no_grad():
            enc = self._tok(
                premises,
                [hypothesis] * len(premises),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)
            probs = torch.softmax(self._model(**enc).logits, dim=-1).cpu().tolist()
        return [
            NLIScore(
                entailment=p[self._idx["entailment"]],
                neutral=p[self._idx["neutral"]],
                contradiction=p[self._idx["contradiction"]],
            )
            for p in probs
        ]

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

    @staticmethod
    def _best(scores: list[NLIScore]) -> NLIScore:
        return max(scores, key=lambda s: (s.entailment, -s.contradiction), default=NLIScore.zero())
