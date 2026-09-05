"""Hosted fallback generator via HF Inference Providers. (Impl: Phase 2.)

Used only when local generation exceeds the timeout. Requires ``HF_TOKEN``
(Kai's is HF Pro, which also lifts the Phase 3 judge's rate limits). Model id and
provider come from config (``AD_FONTES_HOSTED_MODEL`` / ``_PROVIDER``). The same
JSON schema is requested via ``response_format`` so downstream verification is
identical regardless of which backend answered.

If ``HF_TOKEN`` is unset, the fallback is disabled and a local-generation failure
surfaces as an ``error`` event — never a silent switch to a different model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.api.schemas import Audience
from app.generation.local_llm import GenerationChunk
from app.generation.schema import AnswerDraft


class HostedGenerator:  # pragma: no cover - Phase 2
    def __init__(self, model: str, provider: str, token: str) -> None:
        raise NotImplementedError("Implemented in Phase 2 (generation + verification).")

    async def stream(
        self, question: str, context_block: str, audience: Audience
    ) -> AsyncIterator[GenerationChunk]: ...

    def parse(self, raw: str) -> AnswerDraft: ...
