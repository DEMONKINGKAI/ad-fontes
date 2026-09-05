from __future__ import annotations

from pathlib import Path

from app.ingestion.loader import chunk_markdown, load_corpus, slugify

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def test_slugify_rules():
    assert slugify("One-line summary") == "one-line-summary"
    assert slugify("Feature engineering (422,458 × 143)") == "feature-engineering-422-458-143"
    assert slugify("  Trailing / leading --- ") == "trailing-leading"
    assert len(slugify("x" * 200)) <= 80


def test_load_corpus_is_deterministic():
    a = load_corpus(_CORPUS)
    b = load_corpus(_CORPUS)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.embed_text for c in a] == [c.embed_text for c in b]


def test_chunk_ids_unique_and_well_formed():
    chunks = load_corpus(_CORPUS)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for cid in ids:
        stem, _, slug = cid.partition("#")
        assert stem and slug
        assert slug == slug.lower()
        assert " " not in cid


def test_readme_is_excluded():
    chunks = load_corpus(_CORPUS)
    assert not any(c.source_path.endswith("README.md") for c in chunks)
    assert not any("ingestion notes" in c.text.lower() for c in chunks)


def test_breadcrumb_not_in_citable_text_but_in_embed_text():
    chunks = load_corpus(_CORPUS)
    fons = next(c for c in chunks if c.chunk_id == "fons-iuris#one-line-summary")
    assert fons.title == "fons iuris › One-line summary"
    assert "›" not in fons.text
    assert not fons.text.startswith("fons iuris")
    assert fons.embed_text == f"{fons.title}\n{fons.text}"


def test_project_frontmatter_becomes_metadata():
    chunks = load_corpus(_CORPUS)
    fons = next(c for c in chunks if c.chunk_id.startswith("fons-iuris#"))
    assert fons.doc_type == "project"
    assert fons.project_id == "fons-iuris"
    assert fons.repo_url == "https://github.com/DEMONKINGKAI/fons-iuris"
    assert "chromadb" in {s.lower() for s in fons.stack}
    meta = fons.to_chroma_metadata()
    assert isinstance(meta["stack"], str)
    assert "|chromadb|" in meta["stack"]


def test_loan_approval_null_repo_is_none():
    chunks = load_corpus(_CORPUS)
    loan = next(c for c in chunks if c.chunk_id.startswith("loan-approval-kaggle#"))
    assert loan.repo_url is None
    assert loan.project_id == "loan-approval"


def test_faq_chunked_per_question():
    chunks = [c for c in load_corpus(_CORPUS) if c.source_path.endswith("recruiter-faq.md")]
    assert len(chunks) >= 10
    assert all(c.doc_type == "faq" for c in chunks)
    sid = "recruiter-faq#what-is-kai-s-strongest-area"
    strongest = next(c for c in chunks if c.chunk_id == sid)
    assert strongest.text.startswith("Q: What is Kai's strongest area?")
    assert "Causal inference" in strongest.text
    assert strongest.title == "Recruiter FAQ › What is Kai's strongest area"


def test_stack_map_chunked_on_single_hash():
    chunks = [c for c in load_corpus(_CORPUS) if c.source_path.endswith("tech-stack-map.md")]
    sections = {c.section for c in chunks}
    assert sections == {
        "Technology → project map",
        "Datasets and external sources used",
        "Recurring architectural patterns",
    }
    assert all(c.doc_type == "stack_map" for c in chunks)


def test_profile_preamble_becomes_a_chunk():
    chunks = load_corpus(_CORPUS)
    who = next((c for c in chunks if c.chunk_id == "kai-profile#who-kai-is"), None)
    assert who is not None
    assert "goes by" in who.text
    assert not who.text.lstrip().startswith("#")  # the H1 line was stripped


def test_sections_split_on_h2():
    chunks = chunk_markdown(_CORPUS / "projects" / "threadfall.md", _CORPUS)
    ids = {c.chunk_id for c in chunks}
    assert "threadfall#one-line-summary" in ids
    assert "threadfall#key-decisions-and-why" in ids
    # the H1 title line is not a chunk
    assert not any(c.text.strip() == "# Threadfall — The Shattered Pact" for c in chunks)


def test_every_chunk_has_nonempty_text():
    for c in load_corpus(_CORPUS):
        assert c.text.strip()
        assert c.embed_text.strip()
