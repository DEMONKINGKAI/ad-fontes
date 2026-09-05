"""Metric functions for the eval harness. Pure, unit-tested. (Expanded per phase.)

Phase 1: ``hit_at_k`` (chunk- and file-level).
Phase 2: label distribution, ``unsupported_plus_fabricated_per_100``, numeric
         violation rate, latency percentiles.
Phase 5: bootstrap confidence intervals, judge win rate.
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> bool:
    """True if any gold id appears in the first ``k`` retrieved ids."""
    if not gold:
        return True  # nothing to find (negative control): trivially satisfied
    return bool(set(retrieved[:k]) & set(gold))


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 1.0
    found = set(retrieved[:k]) & set(gold)
    return len(found) / len(set(gold))


def per_100(count: int, n_answers: int) -> float:
    """Rate expressed per 100 answers — the brief's headline unit."""
    if n_answers == 0:
        return 0.0
    return 100.0 * count / n_answers
