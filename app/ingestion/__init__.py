"""Corpus ingestion: frontmatter + ``##`` chunking into a Chroma-ready form.

Implemented in Phase 1. The loader is deliberately dependency-light (frontmatter
parsing + string handling) so chunking determinism can be unit-tested without
loading an embedder.
"""
