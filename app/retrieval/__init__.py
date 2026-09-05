"""Dense retrieval over the corpus: embedder, Chroma store, and the retriever.

Implemented in Phase 1. Split three ways so the embedding model is swappable
(brief §3), the vector store is isolated behind a small interface, and the
retriever's filter/rewrite logic is unit-testable with a fake store.
"""
