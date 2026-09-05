from __future__ import annotations

from pathlib import Path

import pytest
from app.ingestion.loader import load_corpus
from app.retrieval.retriever import RetrievalConfig, Retriever
from tests._fakes import HashingEmbedder, InMemoryVectorStore, build_fake_retriever

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(_CORPUS)


@pytest.fixture
def retriever(chunks):
    return build_fake_retriever(chunks)


# --- query analysis -----------------------------------------------------


def test_tech_vocab_built_from_frontmatter(retriever):
    assert "chromadb" in retriever._tech_terms
    assert "causal-learn" in retriever._tech_terms
    assert "pgmpy" in retriever._tech_terms
    # generic terms are excluded from the *boost* set but stay in the vocab
    assert "python" in retriever._tech_terms
    assert "python" not in retriever._boost_terms


def test_detects_projects_by_id_and_name(retriever):
    assert retriever.projects_in_query("tell me about fons iuris") == {"fons-iuris"}
    assert retriever.projects_in_query("what is threadfall?") == {"threadfall"}
    assert retriever.projects_in_query("the loan approval dataset") == {"loan-approval"}
    assert retriever.projects_in_query("what did kai build") == set()


def test_detects_technologies(retriever):
    assert "chromadb" in retriever.techs_in_query("which projects use chromadb")
    assert "pgmpy" in retriever.techs_in_query("where has kai used pgmpy?")
    assert retriever.techs_in_query("what is kai's favourite colour") == set()


# --- ranking + boosts ------------------------------------------------


def test_returns_top_k(retriever):
    out = retriever.retrieve("causal inference and bayesian networks", top_k=6)
    assert len(out) == 6
    assert out == sorted(out, key=lambda r: (-r.score, r.chunk_id))


def test_project_query_surfaces_that_project(retriever):
    out = retriever.retrieve("what is fons iuris and how does its verification work?")
    assert any(r.project_id == "fons-iuris" for r in out[:3])
    assert any("project" in r.boosts for r in out if r.project_id == "fons-iuris")


def test_tech_query_pulls_in_the_stack_map(retriever):
    out = retriever.retrieve("which projects has Kai used ChromaDB in?")
    assert any(r.doc_type == "stack_map" for r in out)
    stack_hit = next(r for r in out if r.doc_type == "stack_map")
    assert "stackmap" in stack_hit.boosts


def test_boosts_can_be_disabled(chunks):
    r = build_fake_retriever(chunks, RetrievalConfig(enable_boosts=False))
    out = r.retrieve("which projects has Kai used ChromaDB in?")
    assert all(r_.boosts == () for r_ in out)
    assert all(r_.score == r_.base_score for r_ in out)


def test_generic_tech_does_not_trigger_stackmap(retriever):
    # 'python' is generic -> no forced stack-map inclusion, no stackmap boost
    out = retriever.retrieve("does kai write python")
    assert not any("stackmap" in r.boosts for r in out)


def test_concept_trigger_detected(retriever):
    assert retriever._has_concept_trigger("which of kai's projects use a vector database?")
    assert retriever._has_concept_trigger("what datasets has kai worked with")
    assert not retriever._has_concept_trigger("what is fons iuris")


def test_concept_trigger_pulls_stackmap_without_a_named_tech(retriever):
    # no concrete tech named, but "vector database" is a concept trigger
    out = retriever.retrieve("which of Kai's projects use a vector database?", top_k=94)
    stack = [r for r in out if r.doc_type == "stack_map"]
    assert stack and any("stackmap" in r.boosts for r in stack)


def test_faq_chunks_are_penalised(retriever):
    # a query whose FAQ chunk does NOT also earn the section-title boost
    out = retriever.retrieve("has Kai shipped models to production?", top_k=20)
    faq = [r for r in out if r.doc_type == "faq"]
    assert faq and all("faq-penalty" in r.boosts for r in faq)
    assert all(r.score < r.base_score for r in faq if "section" not in r.boosts)


def test_section_title_overlap_boost(retriever):
    out = retriever.retrieve("what is Kai's reinforcement learning experience?", top_k=20)
    rl = next(r for r in out if r.chunk_id == "skills#reinforcement-learning")
    assert "section" in rl.boosts


def test_stale_index_rows_are_skipped(chunks):
    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    store.upsert(
        ids=["ghost#gone"],
        embeddings=embedder.embed_documents(["some orphaned text about causal inference"]),
        documents=["orphan"],
        metadatas=[{"doc_type": "project"}],
    )
    r = Retriever(embedder, store, chunks)
    assert r.retrieve("causal inference") == []


def test_retrieved_chunk_carries_citation_fields(retriever):
    out = retriever.retrieve("how does fons iuris evaluate retrieval?")
    top = out[0]
    assert top.chunk_id and top.title and top.section
    assert top.source_path.endswith(".md")
    assert 0.0 <= top.base_score <= 2.0
