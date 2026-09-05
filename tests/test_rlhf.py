from __future__ import annotations

import random
from pathlib import Path

from app.rlhf._io import append_jsonl, done_ids, read_jsonl, write_jsonl
from app.rlhf.gen_questions import build_questions
from app.rlhf.perturb import PerturbationType, apply

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


# --- _io --------------------------------------------------------------


def test_jsonl_roundtrip_and_resume(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"id": "a", "v": 1}, {"id": "b", "v": 2}])
    assert read_jsonl(p) == [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    append_jsonl(p, {"id": "c", "v": 3})
    assert done_ids(p) == {"a", "b", "c"}
    assert read_jsonl(tmp_path / "missing.jsonl") == []


# --- gen_questions ------------------------------------------------


def test_question_set_is_stratified_and_has_holdout():
    rows = build_questions(_CORPUS, target=300)
    assert 200 <= len(rows) <= 300
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    buckets = {r["bucket"] for r in rows}
    assert {
        "project_overview",
        "project_detail",
        "profile",
        "adversarial",
        "negative_control",
    } <= buckets
    holdout = [r for r in rows if r["holdout"]]
    assert 0.1 < len(holdout) / len(rows) < 0.35
    # every project appears
    pids = {r["project_id"] for r in rows if r["project_id"]}
    assert len(pids) >= 7


def test_question_generation_is_deterministic():
    a = build_questions(_CORPUS, target=200, seed=0)
    b = build_questions(_CORPUS, target=200, seed=0)
    assert [r["question"] for r in a] == [r["question"] for r in b]


# --- perturb ---------------------------------------------------


def _rng():
    return random.Random(0)


def test_inflate_number():
    prose = "fons iuris reached 97.2% article-level hit rate."
    out, _claims, applied = apply(PerturbationType.INFLATE_NUMBER, prose, [], _rng())
    assert applied
    assert "97.2%" not in out
    assert "over" in out or "more than" in out


def test_inflate_number_noop_without_a_number():
    out, _c, applied = apply(
        PerturbationType.INFLATE_NUMBER, "Kai builds grounded systems.", [], _rng()
    )
    assert applied is False
    assert out == "Kai builds grounded systems."


def test_upgrade_verb_hits_prose_and_claims():
    prose = "Kai contributed to the EffiGO document pipeline."
    claims = [{"text": "Kai contributed to the pipeline.", "cite": ["a#b"]}]
    out, new_claims, applied = apply(PerturbationType.UPGRADE_VERB, prose, claims, _rng())
    assert applied
    assert "led" in out
    assert "led" in new_claims[0]["text"]


def test_invent_demo_url():
    out, _c, applied = apply(
        PerturbationType.INVENT_DEMO_URL,
        "Threadfall is a narrative RPG.",
        [],
        _rng(),
        project_slug="threadfall",
    )
    assert applied and "http" in out and "demo" in out.lower()


def test_add_unsupported_tech():
    out, _c, applied = apply(
        PerturbationType.ADD_UNSUPPORTED_TECH, "fons iuris is a RAG system.", [], _rng()
    )
    assert applied and "production" in out


def test_first_person_rewrite():
    prose = "Kai built fons iuris. Kai's design keeps the LLM constrained."
    out, claims, applied = apply(
        PerturbationType.FIRST_PERSON,
        prose,
        [{"text": "Kai built fons iuris.", "cite": ["a#b"]}],
        _rng(),
    )
    assert applied
    assert "Kai" not in out
    assert out.startswith("I built")
    assert "Kai" not in claims[0]["text"]


def test_drop_limitation():
    prose = "The model hit 98% accuracy. However, this partly reflects the synthetic generator."
    out, _c, applied = apply(PerturbationType.DROP_LIMITATION, prose, [], _rng())
    assert applied
    assert "However" not in out
    assert "98% accuracy" in out


# --- build_pairs (synthetic candidates/judged jsonl) ----------------


def _seed_rlhf(monkeypatch, tmp_path):
    from app.rlhf import _io, build_pairs

    d = tmp_path / "rlhf"
    d.mkdir()
    monkeypatch.setattr(_io, "RLHF_DIR", d)
    monkeypatch.setattr(build_pairs, "RLHF_DIR", d)
    return d


def test_build_pairs_picks_faithful_over_perturbed(monkeypatch, tmp_path):
    d = _seed_rlhf(monkeypatch, tmp_path)
    from app.rlhf import build_pairs
    from app.rlhf._io import write_jsonl

    retrieved = [{"chunk_id": "a#b", "title": "T", "text": "text", "source_path": "p.md"}]
    write_jsonl(
        d / "candidates.jsonl",
        [
            {
                "question_id": "q1",
                "question": "What is X?",
                "bucket": "project_overview",
                "audience": "recruiter",
                "holdout": False,
                "retrieved": retrieved,
                "candidates": [
                    {
                        "candidate_id": "q1-c0",
                        "source": "hosted",
                        "perturbation": None,
                        "prose": "X is a grounded system.",
                        "verification": {
                            "claims": [
                                {"text": "X is grounded.", "cite": ["a#b"], "label": "supported"}
                            ]
                        },
                    },
                    {
                        "candidate_id": "q1-c1",
                        "source": "perturb:invent_demo_url",
                        "perturbation": "invent_demo_url",
                        "prose": "X is a grounded system. Live demo at http://fake.example.",
                        "verification": {
                            "claims": [
                                {"text": "X is grounded.", "cite": ["a#b"], "label": "supported"}
                            ]
                        },
                    },
                ],
            }
        ],
    )
    write_jsonl(
        d / "judged.jsonl",
        [
            {
                "candidate_id": "q1-c0",
                "question_id": "q1",
                "source": "hosted",
                "perturbation": None,
                "rubric": {},
                "verification": {
                    "unsupported_plus_fabricated": 0,
                    "contradicted": 0,
                    "fabricated": 0,
                    "unverified_prose_count": 0,
                },
                "judge_scalar": 0.9,
                "veto": None,
            },
            {
                "candidate_id": "q1-c1",
                "question_id": "q1",
                "source": "perturb:invent_demo_url",
                "perturbation": "invent_demo_url",
                "rubric": {},
                "verification": {
                    "unsupported_plus_fabricated": 0,
                    "contradicted": 0,
                    "fabricated": 0,
                    "unverified_prose_count": 1,
                },
                "judge_scalar": 0.4,
                "veto": None,
            },
        ],
    )

    stats = build_pairs.build(margin=0.1, seed=0)
    assert stats["pairs_kept"] == 1
    pairs = list(build_pairs.iter_jsonl(d / "pairs_full.jsonl"))
    assert pairs[0]["chosen_source"] == "hosted"
    assert pairs[0]["rejected_perturbation"] == "invent_demo_url"
