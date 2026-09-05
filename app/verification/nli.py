"""Layer 2: NLI entailment check.

``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` cross-encoder, loaded once, run
once per answer on (premise = cited chunk text, hypothesis = claim). Per the
Phase 0 decision NLI runs only on the final claims pass, not on every retrieved
chunk.

Long premises are split on sentence boundaries and scored window-by-window
(fons-iuris found blind token windows produced spurious contradiction scores);
when a claim cites several chunks the highest-entailment window wins. The label
order is read from ``model.config.id2label`` rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9])")
_MAX_WINDOW_CHARS = 600
_MAX_WINDOWS_PER_PREMISE = 5  # bound CPU cost on long chunks
_MAX_PAIRS_PER_CLAIM = 12


@dataclass(slots=True, frozen=True)
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float

    @classmethod
    def zero(cls) -> NLIScore:
        return cls(0.0, 1.0, 0.0)


def _windows(premise: str) -> list[str]:
    premise = premise.strip()
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
    # overlap adjacent windows by one sentence so a claim spanning a boundary is seen
    from itertools import pairwise

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
        return self._best(self._score_pairs(_windows(premise), hypothesis))

    def score_claim(self, claim: str, premises: list[str]) -> NLIScore:
        """Score a claim against the cited chunks.

        The claim is split into sentences (base-model claims are often run-ons —
        "…layer 1 does X. It reverted Y.") and each sentence is scored against
        every premise window. The claim's entailment is the *weakest* sentence
        (all parts must be supported) and its contradiction is the *strongest*
        (any contradicted part taints the claim).
        """
        windows: list[str] = []
        for p in premises:
            windows.extend(_windows(p))
        windows = windows[:_MAX_PAIRS_PER_CLAIM]
        if not windows:
            return NLIScore.zero()

        sentences = (s.strip() for s in _SENT_SPLIT.split(claim.strip()))
        parts = [s for s in sentences if len(s) > 8] or [claim]
        per_part = [self._best(self._score_pairs(windows, part)) for part in parts]
        return NLIScore(
            entailment=min(s.entailment for s in per_part),
            neutral=max(s.neutral for s in per_part),
            contradiction=max(s.contradiction for s in per_part),
        )

    @staticmethod
    def _best(scores: list[NLIScore]) -> NLIScore:
        # the window that most supports the claim; ties broken by lower contradiction
        return max(scores, key=lambda s: (s.entailment, -s.contradiction), default=NLIScore.zero())
