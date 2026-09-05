"""Text -> vector. Swappable behind the ``Embedder`` protocol. (Impl: Phase 1.)

Default model is ``nomic-ai/nomic-embed-text-v1.5`` (brief §3: Kai has measured
it before). nomic requires task-prefixes (``search_document:`` / ``search_query:``)
and ``trust_remote_code``; that lives here so callers just pass text.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def load_embedder(model_name: str) -> Embedder:  # pragma: no cover - Phase 1
    raise NotImplementedError("Implemented in Phase 1 (ingestion + retrieval).")
