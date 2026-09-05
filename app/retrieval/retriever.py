"""Query -> top-k chunks, with metadata filters and optional rewrite. (Impl: Phase 1.)

Behaviour (brief §4):
  * dense top-k (default k=6), return ``chunk_id``s;
  * when a query token matches a known ``project_id`` or a ``stack`` entry, boost
    / filter on that metadata;
  * always add ``stack/tech-stack-map.md`` chunks to the candidate set when the
    query names a technology;
  * optional LLM query rewrite (kept behind a flag; the corpus is ~60 chunks, so
    this is only turned on if the Phase 1 eval justifies it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    section: str
    doc_type: str
    source_path: str
    project_id: str | None
    repo_url: str | None
    score: float


class Retriever:  # pragma: no cover - Phase 1
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        raise NotImplementedError("Implemented in Phase 1 (ingestion + retrieval).")
