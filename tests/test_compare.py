from __future__ import annotations

from pathlib import Path

from app.eval.run_eval import load_questions
from app.ingestion.loader import load_corpus
from app.rlhf.compare import _bootstrap_delta, _eval_questions, _metric, _summary

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
_ADV = Path(__file__).resolve().parent.parent / "app" / "eval" / "adversarial.jsonl"


# --- adversarial set integrity --------------------------------------


def test_adversarial_set_is_well_formed():
    rows = load_questions(_ADV)
    assert len(rows) >= 25
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    subtypes = {r["subtype"] for r in rows}
    assert {"fake_deployment", "verb_inflation", "metric_inflation", "false_premise"} <= subtypes
    # most are unanswerable overclaim-bait
    assert sum(1 for r in rows if not r["answerable"]) >= len(rows) * 0.6

    valid = {c.chunk_id for c in load_corpus(_CORPUS)}
    for r in rows:
        assert r["expected"], f"{r['id']} has no expected-behaviour note"
        for cid in r["gold_chunk_ids"]:
            assert cid in valid, f"{r['id']} cites unknown chunk {cid}"


def test_false_premise_questions_target_real_corpus_contradictions():
    rows = {r["id"]: r for r in load_questions(_ADV)}
    # the corpus README flags: pharmacausal uses causal-learn, not pgmpy
    assert "pgmpy" in rows["adv-pgmpy-pharma"]["question"]
    # Threadfall narrator is Qwen2.5, not Qwen3
    assert "Qwen3" in rows["adv-thread-qwen3"]["question"]


# --- compare metrics -------------------------------------------------


def _row(rid, *, answerable=True, labels=(), declined=False, numeric=0, prose="x", uprose=()):
    return {
        "id": rid,
        "answerable": answerable,
        "labels": list(labels),
        "n_claims": len(labels),
        "numeric_flags": numeric,
        "declined": declined,
        "prose_chars": len(prose),
        "unverified_prose": list(uprose),
        "latency_ms": 1000,
        "generator": "local-base",
    }


def test_summary_counts_per_100():
    rows = [
        _row("a", labels=["supported"]),
        _row("b", labels=["unsupported", "fabricated_citation"]),
        _row("c", labels=["contradicted"], numeric=1),
        _row("d", answerable=False, declined=True),
    ]
    s = _summary(rows)
    assert s["unsupported_plus_fabricated_per_100"] == 50.0  # 2 of 4 answers
    assert s["contradicted_per_100"] == 25.0
    assert s["numeric_violations_per_100"] == 25.0
    assert s["decline_on_unanswerable"] == 1.0
    assert s["false_decline_on_answerable"] == 0.0


def test_metric_matches_summary_for_a_subset():
    rows = [_row("a", labels=["unsupported"]), _row("b", labels=["supported"])]
    assert _metric(rows, "unsupported_plus_fabricated_per_100") == 50.0


def test_bootstrap_delta_direction_and_ci():
    # arm A: every answer has an unsupported claim; arm B: none do
    a = [_row(str(i), labels=["unsupported"]) for i in range(30)]
    b = [_row(str(i), labels=["supported"]) for i in range(30)]
    out = _bootstrap_delta(a, b, "unsupported_plus_fabricated_per_100", n_boot=200, seed=0)
    assert out["delta"] == -100.0  # B is 100/100 better
    lo, hi = out["ci95"]
    assert lo <= -100.0 <= hi <= 0


def test_eval_questions_pulls_adversarial_and_negatives():
    qs = _eval_questions()
    sets = {q["set"] for q in qs}
    assert "adversarial" in sets
    assert "negative_control" in sets
    assert len(qs) >= 30
