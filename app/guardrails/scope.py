"""Decide whether a question is answerable from the corpus.

Two gates (brief §2, §4):
  1. Denylist — topics Kai will never let the assistant speculate on regardless of
     retrieval: salary/compensation, personal/family life, immigration/visa
     status, religion, politics, health. Active now.
  2. Similarity gate — cosine of the query embedding against the corpus centroid;
     below a tuned threshold the question is off-topic ("does Kai know Rust")
     even if it is otherwise innocuous. Needs the embedder -> Phase 2.

Declines are logged (without full IPs) so the Phase 5 negative-control eval can
check the decline rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def denylist_verdict(question: str) -> ScopeVerdict:
    """Fast, model-free check. Returns in_scope=True unless a deny pattern matches."""
    for category, pattern in _DENY_PATTERNS:
        if pattern.search(question):
            return ScopeVerdict(in_scope=False, reason=category)
    return ScopeVerdict(in_scope=True, reason="ok")


def classify_scope(
    question: str, query_embedding=None, centroid=None
) -> ScopeVerdict:  # pragma: no cover - Phase 2
    """Denylist + similarity gate. Phase 2 supplies the embedding + centroid."""
    deny = denylist_verdict(question)
    if not deny.in_scope:
        return deny
    raise NotImplementedError("Similarity gate implemented in Phase 2.")
