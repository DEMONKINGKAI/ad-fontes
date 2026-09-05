"""Decide whether a question is answerable from the corpus.

Two gates (brief §2, §4):
  1. **Denylist** — topics Kai will never let the assistant speculate on
     regardless of retrieval: compensation, personal/family life, immigration,
     politics/health. Model-free, always on. This is the reliable gate.
  2. **Retrieval-floor gate** — decline only when the best retrieved chunk scores
     below ``top_score_threshold`` (default 0.55): the corpus has nothing close,
     so the question is off-distribution (gibberish, a different person, a topic
     with zero coverage).

The centroid signal the brief suggests was measured and dropped: the corpus is
about one person, so every "Kai …" question sits ~0.59–0.62 from the centroid
whether answerable or not (Phase 2 note). "Answerable-looking but unanswerable"
questions ("Does Kai know Rust?") are NOT the gate's job — the generator must say
"the corpus doesn't cover that" and the NLI layer flags it if the model instead
rambles. Declines are logged (no full IPs) for the negative-control metric.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("ad_fontes.scope")

_DENY_TERMS: dict[str, tuple[str, ...]] = {
    "compensation": (
        r"salar(?:y|ies)",
        "compensation",
        "pay",
        "wage",
        "ctc",
        r"how much .* (?:earn|make)",
        r"day ?rate",
        "rate expectation",
    ),
    "personal_life": (
        "girlfriend",
        "boyfriend",
        "partner",
        "marri(?:ed|age)",
        "wife",
        "husband",
        "dating",
        "family",
        "parents",
        "kids",
        "children",
        "religio",
        "caste",
    ),
    "immigration": (
        "visa",
        "immigration",
        "work permit",
        "residence permit",
        "citizenship",
        "green card",
        r"sponsor(?:ship)?",
        "blue card",
    ),
    "politics_health": (
        "politic(?:al|s)",
        r"vote[ds]?",
        "medical condition",
        "mental health",
        "disabilit",
    ),
}

_DENY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (category, re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.I))
    for category, terms in _DENY_TERMS.items()
)


@dataclass(slots=True, frozen=True)
class ScopeVerdict:
    in_scope: bool
    reason: str  # "ok" | "<deny-category>" | "off_topic"

    @property
    def declined(self) -> bool:
        return not self.in_scope


def denylist_verdict(question: str) -> ScopeVerdict:
    for category, pattern in _DENY_PATTERNS:
        if pattern.search(question):
            return ScopeVerdict(in_scope=False, reason=category)
    return ScopeVerdict(in_scope=True, reason="ok")


@dataclass(slots=True, frozen=True)
class ScopeGate:
    centroid: np.ndarray | None = None  # kept for diagnostics; not used to decide
    top_score_threshold: float = 0.55
    centroid_threshold: float = 0.60

    def classify(
        self,
        question: str,
        *,
        query_embedding: np.ndarray | None = None,
        top_score: float | None = None,
    ) -> ScopeVerdict:
        deny = denylist_verdict(question)
        if deny.declined:
            log.info("decline: %s", deny.reason)
            return deny

        if top_score is not None and top_score < self.top_score_threshold:
            log.info("decline: off_topic (top=%.3f < %.2f)", top_score, self.top_score_threshold)
            return ScopeVerdict(in_scope=False, reason="off_topic")

        return ScopeVerdict(in_scope=True, reason="ok")

    def centroid_similarity(self, query_embedding: np.ndarray) -> float | None:
        """Diagnostic only — logged in eval, not part of the decision (see module docstring)."""
        if self.centroid is None:
            return None
        q = np.asarray(query_embedding, dtype=np.float32)
        denom = (np.linalg.norm(q) or 1.0) * (np.linalg.norm(self.centroid) or 1.0)
        return float(q @ self.centroid / denom)
