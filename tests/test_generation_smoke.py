"""Real-model generation smoke test — `-m smoke`, skipped without the deps and a
built index. It is a wiring/regression guard, not the eval (the numbers live in
`python -m app.eval.run_eval --stage generation`).

It runs a handful of questions through the *whole* pipeline (retriever + base
GGUF + NLI + numeric + scope + hosted fallback) and asserts structural
properties, never exact wording.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("sentence_transformers")

from app.api.schemas import AskRequest, ClaimLabel, GeneratorKind  # noqa: E402
from app.config import get_settings  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


@pytest.fixture(scope="module")
def pipeline():
    from app.bootstrap import load_all
    from app.pipeline import Pipeline
    from app.retrieval.retriever import build_retriever

    s = get_settings()
    if build_retriever(s).index_health()["indexed"] == 0:
        pytest.skip("no Chroma index (run: python -m app.ingestion.cli --rebuild)")
    return Pipeline(s, load_all(s, load_tuned=False))


async def test_grounded_answer_has_verified_claims(pipeline):
    resp = await pipeline.answer_sync(
        AskRequest(question="What is fons iuris?", audience="recruiter")
    )
    assert resp.declined is False
    assert resp.prose and "fons" in resp.prose.lower()
    assert resp.sources and resp.meta.retrieved_chunk_ids
    assert resp.meta.generator in {
        GeneratorKind.local_base,
        GeneratorKind.hosted_fallback,
    }
    # every claim's citations resolve to retrieved chunks (or are flagged)
    retrieved = set(resp.meta.retrieved_chunk_ids)
    for c in resp.claims:
        if c.verification.label is not ClaimLabel.fabricated_citation:
            assert any(cid in retrieved for cid in c.cite)


async def test_denylist_question_is_declined_without_generating(pipeline):
    resp = await pipeline.answer_sync(AskRequest(question="What is Kai's salary?"))
    assert resp.declined is True
    assert resp.claims == []
    assert resp.meta.in_scope is False
    assert resp.meta.generation_ms == 0


async def test_overclaim_is_not_marked_supported(pipeline):
    # the corpus says demo:null for Threadfall; a "deployed to production" claim
    # must not come back 'supported'
    resp = await pipeline.answer_sync(
        AskRequest(question="Did Kai deploy Threadfall to production for real users?")
    )
    for c in resp.claims:
        if "production" in c.text.lower() and "real user" in c.text.lower():
            assert c.verification.label is not ClaimLabel.supported


async def test_streaming_emits_tokens_then_claims(pipeline):
    kinds = []
    async for ev in pipeline.answer_stream(AskRequest(question="What is Threadfall?")):
        kinds.append(ev.event)
    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "done"
