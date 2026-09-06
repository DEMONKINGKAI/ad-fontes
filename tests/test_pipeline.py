from __future__ import annotations

from pathlib import Path

import pytest
from app.api.schemas import AskRequest, ClaimLabel, GeneratorKind, ModelChoice
from app.config import Settings
from app.ingestion.loader import load_corpus
from app.pipeline import Pipeline, PipelineComponents, PipelineNotReadyError
from app.verification.nli import NLIScore
from tests._fakes import FakeGenerator, FakeNLI, FakeRetriever

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(_CORPUS)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        corpus_dir=_CORPUS,
        index_dir=tmp_path,
        feedback_dir=tmp_path,
        local_timeout_s=5.0,
        eager_model_load=False,
    )


_DEFAULT_IDS = [
    "fons-iuris#one-line-summary",
    "fons-iuris#evaluation-methodology-and-results",
    "fons-iuris#pipeline",
    "threadfall#one-line-summary",
    "recruiter-faq#which-project-best-shows-rag-skills",
    "skills#generative-ai-nlp",
]


def _pipeline(settings, chunks, *, generator, nli=None, hosted=None, tuned=None, want_ids=None):
    retriever = FakeRetriever(chunks, want_ids or _DEFAULT_IDS)
    comps = PipelineComponents(
        retriever=retriever,
        embedder=retriever.embedder,
        base_generator=generator,
        tuned_generator=tuned,
        hosted_generator=hosted,
        nli=nli or FakeNLI(),
    )
    return Pipeline(settings, comps)


async def test_not_ready_raises(settings):
    p = Pipeline(settings, PipelineComponents())
    with pytest.raises(PipelineNotReadyError):
        await p.answer_sync(AskRequest(question="What is fons iuris?"))


async def test_happy_path_verifies_claims(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose="Kai built fons iuris, a grounded RAG system.",
        claims=[
            {
                "text": "Kai built fons iuris, a grounded RAG system.",
                "cite": ["fons-iuris#one-line-summary"],
            },
        ],
    )
    p = _pipeline(settings, chunks, generator=gen)
    resp = await p.answer_sync(AskRequest(question="What is fons iuris?", audience="recruiter"))

    assert resp.declined is False
    assert resp.prose.startswith("Kai built fons iuris")
    assert len(resp.claims) == 1
    assert resp.claims[0].verification.label is ClaimLabel.supported
    assert resp.claims[0].sources[0].chunk_id == "fons-iuris#one-line-summary"
    assert resp.meta.generator is GeneratorKind.local_base
    assert resp.meta.retrieved_chunk_ids
    assert resp.sources


async def test_fabricated_citation_is_caught(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose="Kai deployed it to 10,000 users.",
        claims=[{"text": "Kai deployed it to 10000 users.", "cite": ["ghost#not-retrieved"]}],
    )
    p = _pipeline(settings, chunks, generator=gen)
    resp = await p.answer_sync(AskRequest(question="How many users does fons iuris have?"))
    assert resp.claims[0].verification.label is ClaimLabel.fabricated_citation


async def test_contradiction_label(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose="fons iuris hides hallucinations.",
        claims=[
            {"text": "fons iuris hides hallucinations.", "cite": ["fons-iuris#one-line-summary"]}
        ],
    )
    nli = FakeNLI(rules={"hides hallucinations": (0.02, 0.1, 0.88)})
    p = _pipeline(settings, chunks, generator=gen, nli=nli)
    resp = await p.answer_sync(AskRequest(question="Does fons iuris hide hallucinations?"))
    assert resp.claims[0].verification.label is ClaimLabel.contradicted


async def test_numeric_guard_flags_inflated_number(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose="fons iuris hits over 99% article-level retrieval.",
        claims=[
            {
                "text": "fons iuris reaches over 99% article-level retrieval hit rate.",
                "cite": ["fons-iuris#evaluation-methodology-and-results"],
            }
        ],
    )
    p = _pipeline(settings, chunks, generator=gen)
    resp = await p.answer_sync(AskRequest(question="How good is fons iuris retrieval?"))
    assert resp.claims[0].verification.numeric_flag is True


async def test_prose_hallucination_is_flagged_even_when_claims_are_safe(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose=(
            "fons iuris is a grounded RAG system. "
            "Kai deployed it to Kubernetes for fifty thousand paying customers."
        ),
        claims=[
            {
                "text": "fons iuris is a grounded RAG system.",
                "cite": ["fons-iuris#one-line-summary"],
            }
        ],
    )
    # the safe claim genuinely entails; only the extra prose sentence is unbacked
    nli = FakeNLI(default=(0.05, 0.9, 0.05), rules={"grounded RAG system": (0.95, 0.03, 0.02)})
    p = _pipeline(settings, chunks, generator=gen, nli=nli)
    resp = await p.answer_sync(AskRequest(question="Tell me about fons iuris deployment"))
    assert any("Kubernetes" in s for s in resp.unverified_prose)
    assert not any("grounded RAG system" in s for s in resp.unverified_prose)


async def test_denylist_declines(settings, chunks):
    gen = FakeGenerator(GeneratorKind.local_base, prose="x", claims=[])
    p = _pipeline(settings, chunks, generator=gen)
    resp = await p.answer_sync(AskRequest(question="What is Kai's expected salary?"))
    assert resp.declined is True
    assert resp.claims == []
    assert resp.meta.in_scope is False
    assert gen.calls == 0  # generator never invoked for a declined question


async def test_timeout_falls_back_to_hosted(settings, chunks):
    slow = FakeGenerator(GeneratorKind.local_base, prose="partial", claims=[], timed_out=True)
    hosted = FakeGenerator(
        GeneratorKind.hosted_fallback,
        prose="Kai built fons iuris.",
        claims=[{"text": "Kai built fons iuris.", "cite": ["fons-iuris#one-line-summary"]}],
    )
    p = _pipeline(settings, chunks, generator=slow, hosted=hosted)
    resp = await p.answer_sync(AskRequest(question="What is fons iuris?"))
    assert resp.meta.generator is GeneratorKind.hosted_fallback
    assert "fons iuris" in resp.prose


async def test_no_hosted_no_silent_swap(settings, chunks):
    slow = FakeGenerator(
        GeneratorKind.local_base, prose="partial answer", claims=[], timed_out=True
    )
    p = _pipeline(settings, chunks, generator=slow, hosted=None)
    resp = await p.answer_sync(AskRequest(question="What is fons iuris?"))
    assert resp.meta.generator is GeneratorKind.local_base  # stays honest


async def test_stream_event_order(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose="Kai built fons iuris.",
        claims=[{"text": "Kai built fons iuris.", "cite": ["fons-iuris#one-line-summary"]}],
    )
    p = _pipeline(settings, chunks, generator=gen)
    events = [e.event async for e in p.answer_stream(AskRequest(question="What is fons iuris?"))]
    assert events[0] == "sources"
    assert "token" in events
    assert events.index("sources") < events.index("token") < events.index("claims")
    assert events.index("claims") < events.index("meta") < events.index("done")


async def test_tuned_requested_but_absent_serves_base(settings, chunks):
    gen = FakeGenerator(
        GeneratorKind.local_base,
        prose="Kai built fons iuris.",
        claims=[{"text": "Kai built fons iuris.", "cite": ["fons-iuris#one-line-summary"]}],
    )
    p = _pipeline(settings, chunks, generator=gen, tuned=None)
    resp = await p.answer_sync(AskRequest(question="What is fons iuris?", model=ModelChoice.tuned))
    assert resp.meta.model_requested is ModelChoice.base  # honest about what ran
    assert resp.meta.generator is GeneratorKind.local_base


def test_nli_score_zero_helper():
    z = NLIScore.zero()
    assert z.entailment == 0.0 and z.neutral == 1.0
