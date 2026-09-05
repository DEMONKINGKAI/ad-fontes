"""Metric functions for the eval harness. Pure, unit-tested.

Phase 1: retrieval hit@k (chunk- and file-level), MRR, recall@k.
Phase 2: label distribution, ``unsupported_plus_fabricated_per_100``, numeric
         violation rate, latency percentiles.
Phase 5: bootstrap confidence intervals, judge win rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> bool:
    """True if any gold id appears in the first ``k`` retrieved ids.

    An empty gold set (negative control) is treated as "nothing to find" and
    returns True — negative controls are scored separately, not via hit@k.
    """
    if not gold:
        return True
    return bool(set(retrieved[:k]) & set(gold))


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def first_rank(retrieved: Sequence[str], gold: Sequence[str]) -> int | None:
    """1-indexed rank of the first gold hit, or None if not retrieved."""
    gold_set = set(gold)
    for i, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            return i
    return None


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    r = first_rank(retrieved, gold)
    return 1.0 / r if r else 0.0


def mrr(ranks: Sequence[float]) -> float:
    return mean(ranks) if ranks else 0.0


def per_100(count: int, n_answers: int) -> float:
    """Rate expressed per 100 answers — the brief's headline unit."""
    if n_answers == 0:
        return 0.0
    return 100.0 * count / n_answers


def rate(hits: Sequence[bool]) -> float:
    return mean(1.0 if h else 0.0 for h in hits) if hits else 0.0
