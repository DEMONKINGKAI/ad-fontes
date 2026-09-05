from __future__ import annotations

import pytest
from app.api.schemas import AskRequest, Audience, ClaimLabel
from app.generation.schema import ANSWER_JSON_SCHEMA, AnswerDraft
from pydantic import ValidationError


def test_ask_request_defaults():
    r = AskRequest(question="What did Kai build?")
    assert r.audience is Audience.auto
    assert r.model is None
    assert r.session_id is None


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_request_accepts_known_audiences():
    assert AskRequest(question="q", audience="engineer").audience is Audience.engineer


def test_answer_draft_tolerates_missing_citation_for_verification_to_flag():
    # the grammar asks for >=1 cite, but the schema accepts an empty list so the
    # structural layer can label an uncited claim `fabricated_citation`
    d = AnswerDraft(prose="x", claims=[{"text": "uncited", "cite": []}])
    assert d.claims[0].cite == []


def test_answer_draft_ok():
    d = AnswerDraft(
        prose="Kai built fons iuris.",
        claims=[{"text": "Kai built fons iuris.", "cite": ["fons-iuris#one-line-summary"]}],
    )
    assert d.claims[0].cite == ["fons-iuris#one-line-summary"]


def test_claim_labels_are_four():
    assert {label.value for label in ClaimLabel} == {
        "supported",
        "unsupported",
        "contradicted",
        "fabricated_citation",
    }


def test_answer_json_schema_shape():
    props = ANSWER_JSON_SCHEMA["properties"]
    assert set(ANSWER_JSON_SCHEMA["required"]) == {"prose", "claims"}
    assert props["claims"]["items"]["required"] == ["text", "cite"]
    assert props["claims"]["items"]["properties"]["cite"]["minItems"] == 1
