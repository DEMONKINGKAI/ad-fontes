from __future__ import annotations

from pathlib import Path

from app.eval import metrics
from app.eval.run_eval import load_questions
from app.ingestion.loader import load_corpus

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


# --- metrics ------------------------------------------------------------


def test_hit_at_k():
    assert metrics.hit_at_k(["a", "b", "c"], ["c"], 3) is True
    assert metrics.hit_at_k(["a", "b", "c"], ["c"], 2) is False
    assert metrics.hit_at_k(["a"], [], 1) is True  # negative control: nothing to find


def test_first_rank_and_reciprocal():
    assert metrics.first_rank(["a", "b", "c"], ["b"]) == 2
    assert metrics.first_rank(["a", "b"], ["z"]) is None
    assert metrics.reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert metrics.reciprocal_rank(["a"], ["z"]) == 0.0


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5


def test_rate_and_per_100():
    assert metrics.rate([True, True, False, False]) == 0.5
    assert metrics.per_100(3, 150) == 2.0
    assert metrics.per_100(0, 0) == 0.0


# --- questions.jsonl integrity ---------------------------------------


def test_question_set_is_well_formed():
    questions = load_questions()
    assert len(questions) >= 40
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    negatives = [q for q in questions if not q["answerable"]]
    assert len(negatives) >= 5  # brief: include negative controls

    valid_ids = {c.chunk_id for c in load_corpus(_CORPUS)}
    valid_files = {c.source_path for c in load_corpus(_CORPUS)}
    for q in questions:
        assert q["category"]
        assert q["question"].strip()
        for cid in q["gold_chunk_ids"]:
            assert cid in valid_ids, f"{q['id']} cites unknown chunk {cid}"
        for f in q["gold_files"]:
            assert f in valid_files, f"{q['id']} cites unknown file {f}"
        if q["answerable"]:
            assert q["gold_chunk_ids"], f"{q['id']} is answerable but has no gold"
        else:
            assert not q["gold_chunk_ids"], f"{q['id']} is a negative control with gold"


def test_negative_controls_cover_the_brief_examples():
    questions = {q["id"]: q for q in load_questions()}
    joined = " ".join(q["question"].lower() for q in questions.values() if not q["answerable"])
    assert "salary" in joined
    assert "rust" in joined
