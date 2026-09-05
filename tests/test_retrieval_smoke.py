"""Real-model smoke test: skipped unless the retrieval deps *and* a built index
are present (i.e. it runs locally after `python -m app.ingestion.cli --rebuild`,
and is skipped in the light CI job).

It is a regression guard, not the evaluation — the numbers live in
`python -m app.eval.run_eval --stage retrieval`.
"""

from __future__ import annotations

import pytest
from app.config import get_settings

pytestmark = pytest.mark.smoke

st = pytest.importorskip("sentence_transformers")
chromadb = pytest.importorskip("chromadb")


@pytest.fixture(scope="module")
def retriever():
    from app.retrieval.retriever import build_retriever

    r = build_retriever(get_settings())
    if r.index_health()["indexed"] == 0:
        pytest.skip("no Chroma index built (run `python -m app.ingestion.cli --rebuild`)")
    return r


def test_index_matches_corpus(retriever):
    h = retriever.index_health()
    assert h["indexed"] == h["expected"]


@pytest.mark.parametrize(
    "question,expect_project",
    [
        ("What is fons iuris and how does it verify claims?", "fons-iuris"),
        ("How does Threadfall keep the LLM from hallucinating?", "threadfall"),
        ("What did pharmacausal find in the FAERS data?", "pharmacausal"),
        ("How large is evidentia's Bayesian network?", "evidentia"),
    ],
)
def test_project_questions_retrieve_that_project(retriever, question, expect_project):
    out = retriever.retrieve(question, top_k=6)
    assert any(r.project_id == expect_project for r in out[:3])


def test_tech_question_surfaces_stack_map(retriever):
    out = retriever.retrieve("which of Kai's projects use ChromaDB?", top_k=6)
    assert any(r.doc_type == "stack_map" for r in out)


def test_negative_control_has_lower_confidence(retriever):
    ans = retriever.retrieve("What is Kai's strongest area?", top_k=1)[0].score
    neg = retriever.retrieve("What is Kai's salary expectation?", top_k=1)[0].score
    assert neg < ans
