from __future__ import annotations

import pytest
from app.api.schemas import Audience
from app.guardrails.audience import resolve_audience
from app.guardrails.scope import denylist_verdict


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
