"""Text -> vector. Swappable behind the ``Embedder`` protocol.

Default model: ``nomic-ai/nomic-embed-text-v1.5`` (brief §3 — Kai has measured it
before). nomic needs task prefixes (``search_document:`` for corpus chunks,
``search_query:`` for questions) and ``trust_remote_code=True``; that is hidden
here so callers just pass plain text. Embeddings are L2-normalised, so a dot
product is cosine similarity and Chroma's ``cosine`` space agrees with our
in-Python re-ranking.

``load_embedder`` is the only entry point. Tests use a deterministic fake that
satisfies the same protocol without downloading a model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract the retriever and ingestion depend on."""

    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of unit-norm row vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a ``(dim,)`` unit-norm vector."""
        ...


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model, applying nomic-style prefixes."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._prefixed = "nomic" in model_name.lower()
        self._model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=16,
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prep = [f"{_DOC_PREFIX}{t}" if self._prefixed else t for t in texts]
        return self._encode(prep)

    def embed_query(self, text: str) -> np.ndarray:
        prep = f"{_QUERY_PREFIX}{text}" if self._prefixed else text
        return self._encode([prep])[0]


def load_embedder(model_name: str, *, device: str = "cpu") -> Embedder:
    return SentenceTransformerEmbedder(model_name, device=device)
