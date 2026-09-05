"""llama-cpp-python backend for the ``base`` and ``tuned`` GGUF generators. (Impl: Phase 2.)

Both models are Q4_K_M GGUF, loaded once at startup (brief §2). Generation is
schema-constrained (``app.generation.schema``); prose tokens are streamed as they
arrive and the JSON is assembled and validated at the end.

A per-request wall-clock budget (``settings.local_timeout_s``) governs the
fallback: if the local model has not finished by then, the caller switches to
``hosted_llm`` and marks the response ``generator="hosted-fallback"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.api.schemas import Audience, ModelChoice
from app.generation.schema import AnswerDraft


@dataclass(slots=True)
class GenerationChunk:
    delta: str
    done: bool = False


class LocalGenerator:  # pragma: no cover - Phase 2
    def __init__(self, which: ModelChoice, model_path: str) -> None:
        raise NotImplementedError("Implemented in Phase 2 (generation + verification).")

    async def stream(
        self, question: str, context_block: str, audience: Audience
    ) -> AsyncIterator[GenerationChunk]: ...

    def parse(self, raw: str) -> AnswerDraft: ...
