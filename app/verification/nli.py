"""Layer 2: NLI entailment check. (Impl: Phase 2.)

``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` cross-encoder, run once per
answer on (premise = cited chunk text, hypothesis = claim). Per the Phase 0
decision, NLI runs only on the final claims pass, not on every retrieved chunk.

Long premises are split on real section/point boundaries before scoring (blind
token windows produced spurious contradiction scores in fons-iuris); when a claim
cites several chunks, the max entailment across them wins.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float


class NLIVerifier:  # pragma: no cover - Phase 2
    def __init__(self, model_name: str) -> None:
        raise NotImplementedError("Implemented in Phase 2 (generation + verification).")

    def score(self, premise: str, hypothesis: str) -> NLIScore: ...

    def score_claim(self, claim: str, premises: list[str]) -> NLIScore: ...
