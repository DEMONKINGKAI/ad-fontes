from __future__ import annotations

import json

import pytest
from app.api.schemas import Audience
from app.generation.base import ProseStreamer, parse_answer
from app.generation.prompts import format_context, system_prompt
from app.generation.schema import ANSWER_JSON_SCHEMA, AnswerDraft
from app.verification.verify import verify_prose

# --- ProseStreamer ----------------------------------------------------


def test_prose_streamer_emits_only_the_prose_value():
    s = ProseStreamer()
    out = ""
    for piece in ['{"pro', 'se": "Kai bui', "lt ", 'fons iuris.", "claims": [', "]}"]:
        out += s.feed(piece)
    assert out == "Kai built fons iuris."
    assert s.prose == "Kai built fons iuris."


def test_prose_streamer_unescapes():
    s = ProseStreamer()
    delta = s.feed(r'{"prose": "line one\nline \"two\"", "claims": []}')
    assert delta == 'line one\nline "two"'


def test_prose_streamer_stops_at_closing_quote():
    s = ProseStreamer()
    s.feed('{"prose": "done", "claims": [{"text": "x", "cite": ["a#b"]}]}')
    assert s.prose == "done"
    assert s.feed("") == ""  # nothing more emitted


# --- parse_answer ----------------------------------------------------


def test_parse_answer_plain():
    d = parse_answer('{"prose": "p", "claims": [{"text": "t", "cite": ["a#b"]}]}')
    assert isinstance(d, AnswerDraft)
    assert d.claims[0].cite == ["a#b"]


def test_parse_answer_with_code_fence_and_prose_around():
    raw = 'Sure!\n```json\n{"prose": "p", "claims": []}\n```\nHope that helps'
    d = parse_answer(raw)
    assert d.prose == "p" and d.claims == []


def test_parse_answer_nested_braces_in_strings():
    raw = '{"prose": "uses {W·g + b}", "claims": [{"text": "t {x}", "cite": ["a#b"]}]}'
    d = parse_answer(raw)
    assert "W·g" in d.prose


def test_parse_answer_no_object_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_answer("the model refused")


# --- prompts -------------------------------------------------------


def test_system_prompt_is_audience_aware_and_third_person():
    rec = system_prompt(Audience.recruiter)
    eng = system_prompt(Audience.engineer)
    assert "third person" in rec
    assert "recruiter" in rec.lower()
    assert "engineer" in eng.lower()
    assert rec != eng


def test_format_context_labels_chunk_ids(chunks_sample):
    block = format_context(chunks_sample)
    assert "id: fons-iuris#one-line-summary" in block
    assert chunks_sample[0].text in block


@pytest.fixture
def chunks_sample():
    from pathlib import Path

    from app.ingestion.loader import load_corpus
    from app.retrieval.retriever import RetrievedChunk

    corpus = Path(__file__).resolve().parent.parent / "data" / "corpus"
    src = {c.chunk_id: c for c in load_corpus(corpus)}
    out = []
    for cid in ["fons-iuris#one-line-summary", "threadfall#one-line-summary"]:
        c = src[cid]
        out.append(
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                title=c.title,
                section=c.section,
                doc_type=c.doc_type,
                source_path=c.source_path,
                project_id=c.project_id,
                repo_url=c.repo_url,
                score=0.8,
                base_score=0.8,
            )
        )
    return out


# --- prose verification -----------------------------------------


def test_verify_prose_flags_sentence_not_in_claims_or_chunks(chunks_sample):
    # a claim covers sentence 1; sentence 2 asserts something unsupported
    from app.api.schemas import Claim, ClaimLabel, ClaimVerification

    claims = [
        Claim(
            text="Threadfall is a solo narrative RPG decided by a causal engine.",
            cite=["threadfall#one-line-summary"],
            verification=ClaimVerification(
                label=ClaimLabel.supported, entailment=0.9, contradiction=0.0
            ),
        )
    ]
    prose = (
        "Threadfall is a solo narrative RPG decided by a causal engine. "
        "Kai deployed it to Kubernetes for ten thousand paying users."
    )
    flagged = verify_prose(prose, claims, chunks_sample, nli=None)
    assert any("Kubernetes" in s for s in flagged)
    assert not any("solo narrative RPG" in s for s in flagged)


def test_verify_prose_flags_sentence_behind_a_failing_claim(chunks_sample):
    """A prose sentence mirrored only by an *unsupported* claim is not covered —
    it goes through NLI and, unbacked, is flagged (the pgmpy/pharmacausal case)."""
    from app.api.schemas import Claim, ClaimLabel, ClaimVerification
    from tests._fakes import FakeNLI

    claims = [
        Claim(
            text="Kai chose pgmpy for the Threadfall engine.",
            cite=["threadfall#one-line-summary"],
            verification=ClaimVerification(
                label=ClaimLabel.unsupported, entailment=0.03, contradiction=0.02
            ),
        )
    ]
    prose = "Kai chose pgmpy for the Threadfall engine because it handles large DAGs well."
    nli = FakeNLI(default=(0.04, 0.92, 0.04))
    flagged = verify_prose(prose, claims, chunks_sample, nli)
    assert flagged == [prose]


# --- schema ------------------------------------------------------


def test_answer_json_schema_still_matches_draft():
    props = ANSWER_JSON_SCHEMA["properties"]
    assert set(props) == {"prose", "claims"}
    d = AnswerDraft.model_validate(
        json.loads('{"prose": "p", "claims": [{"text": "t", "cite": ["a#b"]}]}')
    )
    assert d.claims[0].text == "t"
