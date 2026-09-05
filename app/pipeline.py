"""End-to-end orchestration: question -> guardrails -> retrieval -> generation
-> verification -> response.

This module owns the sequencing described in brief §4 and is the single place the
API layer calls. It is intentionally separate from ``app.api`` so the whole
pipeline can be driven from ``app/eval`` and ``app/rlhf`` scripts without spinning
up FastAPI.

Phase 0: the class and its method signatures exist; ``PipelineNotReadyError`` is raised
until the retrieval (Phase 1) and generation/verification (Phase 2) pieces are
implemented. The API returns 503 with that message rather than a stack trace.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.api.schemas import AskRequest, AskResponse
from app.api.sse import SSEEvent
from app.config import Settings


class PipelineNotReadyError(RuntimeError):
    """Raised while the model-backed stages are still stubs."""


@dataclass(slots=True)
class PipelineComponents:
    """Handles loaded once at startup and injected into the pipeline."""

    retriever: object | None = None
    base_generator: object | None = None
    tuned_generator: object | None = None
    hosted_generator: object | None = None
    nli: object | None = None
    embedder: object | None = None

    @property
    def ready(self) -> bool:
        return all(
            x is not None
            for x in (self.retriever, self.base_generator, self.tuned_generator, self.nli)
        )


class Pipeline:
    def __init__(self, settings: Settings, components: PipelineComponents) -> None:
        self.settings = settings
        self.components = components

    async def answer_sync(self, request: AskRequest) -> AskResponse:  # pragma: no cover - Phase 2
        if not self.components.ready:
            raise PipelineNotReadyError(
                "Generation pipeline is implemented in Phase 2. "
                "Retrieval-only endpoints (/api/projects, /api/health) work now."
            )
        raise NotImplementedError

    async def answer_stream(
        self, request: AskRequest
    ) -> AsyncIterator[SSEEvent]:  # pragma: no cover - Phase 2
        if not self.components.ready:
            raise PipelineNotReadyError(
                "Generation pipeline is implemented in Phase 2. "
                "Retrieval-only endpoints (/api/projects, /api/health) work now."
            )
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator
