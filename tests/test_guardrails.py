from __future__ import annotations

import pytest
from app.api.schemas import Audience
from app.guardrails.audience import resolve_audience
from app.guardrails.scope import ScopeGate, denylist_verdict


@pytest.mark.parametrize(
    "question,category",
    [
        ("What is Kai's expected salary?", "compensation"),
        ("How much does Kai want to make?", "compensation"),
        ("Does Kai have a girlfriend?", "personal_life"),
        ("What is Kai's visa status in Germany?", "immigration"),
        ("Does Kai need visa sponsorship?", "immigration"),
    ],
)
def test_denylist_blocks_out_of_scope(question, category):
    v = denylist_verdict(question)
    assert v.in_scope is False
    assert v.reason == category


@pytest.mark.parametrize(
    "question",
    [
        "Which projects use Bayesian networks?",
        "What did Kai do at EffiGO?",
        "Tell me about fons iuris.",
    ],
)
def test_denylist_allows_in_scope(question):
    assert denylist_verdict(question).in_scope is True


def test_explicit_audience_is_respected():
    assert (
        resolve_audience(Audience.recruiter, "how does the NLI layer work?") is Audience.recruiter
    )


def test_auto_audience_picks_engineer_for_technical_question():
    q = "How does the retrieval pipeline handle chunking and embedding?"
    assert resolve_audience(Audience.auto, q) is Audience.engineer


def test_auto_audience_defaults_to_recruiter():
    q = "Is Kai a good fit for a senior ML role and when can he start?"
    assert resolve_audience(Audience.auto, q) is Audience.recruiter


# --- scope gate --------------------------------------------------------


def test_scope_gate_denylist_wins_over_retrieval():
    gate = ScopeGate()
    v = gate.classify("What is Kai's salary?", top_score=0.99)
    assert v.declined and v.reason == "compensation"


def test_scope_gate_allows_normal_question_with_ok_retrieval():
    gate = ScopeGate(top_score_threshold=0.55)
    assert gate.classify("What did Kai do at EffiGO?", top_score=0.70).in_scope is True


def test_scope_gate_declines_only_on_very_low_retrieval():
    gate = ScopeGate(top_score_threshold=0.55)
    assert gate.classify("asdf qwerty zxcv", top_score=0.30).declined is True
    # an on-distribution but unanswerable question is NOT the gate's job
    assert gate.classify("Does Kai know Rust?", top_score=0.66).in_scope is True


def test_scope_gate_no_top_score_never_declines_on_similarity():
    assert ScopeGate().classify("anything at all", top_score=None).in_scope is True
