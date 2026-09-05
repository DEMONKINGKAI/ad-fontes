"""End-to-end orchestration: question -> guardrails -> retrieval -> generation
-> verification -> response (brief §4).

The single place the API layer calls, and also driven directly by ``app/eval``
and ``app/rlhf`` without FastAPI. Internally ``_run`` is one async generator that
yields ``(kind, payload)`` steps (``token`` / ``sources`` / ``claims`` / ``meta``
/ ``result``); ``answer_stream`` maps them to SSE events and ``answer_sync``
keeps only the final ``result``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.api import sse
from app.api.schemas import (
    AskRequest,
    AskResponse,
    Audience,
    Claim,
    GeneratorKind,
    ModelChoice,
    ResponseMeta,
)
from app.api.sse import SSEEvent
from app.config import Settings
from app.generation.base import GeneratedAnswer, Generator
from app.generation.prompts import DECLINE_MESSAGE, format_context
from app.guardrails.audience import resolve_audience
from app.guardrails.scope import ScopeGate, ScopeVerdict
from app.verification.verify import to_source_chunk, verify_answer


class PipelineNotReadyError(RuntimeError):
    """Raised while the model-backed stages are still loading."""


@dataclass(slots=True)
class PipelineComponents:
    """Handles loaded once at startup and injected into the pipeline."""

    retriever: object | None = None
    embedder: object | None = None
    base_generator: Generator | None = None
    tuned_generator: Generator | None = None
    hosted_generator: Generator | None = None
    nli: object | None = None
    scope_gate: ScopeGate = field(default_factory=ScopeGate)

    @property
    def ready(self) -> bool:
        return (
            self.retriever is not None and self.base_generator is not None and self.nli is not None
        )


@dataclass(slots=True)
class _Timings:
    start: float = field(default_factory=time.monotonic)
    retrieval_ms: int = 0
    generation_ms: int = 0
    verification_ms: int = 0

    def total_ms(self) -> int:
        return int((time.monotonic() - self.start) * 1000)


class Pipeline:
    def __init__(self, settings: Settings, components: PipelineComponents) -> None:
        self.settings = settings
        self.components = components

    # -- helpers -------------------------------------------------------

    def _require_ready(self) -> None:
        if not self.components.ready:
            raise PipelineNotReadyError(
                "Generation pipeline is still loading (embedder / GGUF / NLI). "
                "Retrieval-only endpoints (/api/projects, /api/health) work now."
            )

    def _pick_generator(self, requested: ModelChoice | None) -> tuple[Generator, ModelChoice]:
        c = self.components
        want = requested or ModelChoice(self.settings.default_model)
        if want is ModelChoice.tuned and c.tuned_generator is not None:
            return c.tuned_generator, ModelChoice.tuned
        # tuned not available yet (lands in Phase 4) -> base, reported honestly in meta
        assert c.base_generator is not None
        return c.base_generator, ModelChoice.base

    def _scope(self, question: str, q_vec, top_score: float | None) -> ScopeVerdict:
        return (self.components.scope_gate or ScopeGate()).classify(
            question, query_embedding=q_vec, top_score=top_score
        )

    def _needs_fallback(self, answer: GeneratedAnswer) -> bool:
        d = answer.draft
        return (
            answer.timed_out
            or answer.error is not None
            or not d.prose.strip()
            or (not d.claims and answer.generator is not GeneratorKind.hosted_fallback)
        )

    def _meta(
        self,
        *,
        generator: GeneratorKind,
        requested: ModelChoice,
        audience: Audience,
        retrieved_ids: list[str],
        timings: _Timings,
        in_scope: bool,
    ) -> ResponseMeta:
        return ResponseMeta(
            generator=generator,
            model_requested=requested,
            audience_resolved=audience,
            latency_ms=timings.total_ms(),
            retrieval_ms=timings.retrieval_ms,
            generation_ms=timings.generation_ms,
            verification_ms=timings.verification_ms,
            retrieved_chunk_ids=retrieved_ids,
            corpus_version=self.settings.corpus_version,
            in_scope=in_scope,
        )

    # -- core ---------------------------------------------------------

    async def _run(self, request: AskRequest):
        """Yield ``(kind, payload)`` steps. ``kind`` in
        {token, sources, claims, meta, result}."""
        self._require_ready()
        t = _Timings()
        question = request.question.strip()
        audience = resolve_audience(request.audience, question)
        answer_id = uuid.uuid4().hex
        c = self.components

        q_vec = c.embedder.embed_query(question)  # type: ignore[union-attr]

        r0 = time.monotonic()
        retrieved = c.retriever.retrieve(  # type: ignore[union-attr]
            question, top_k=self.settings.retrieval_top_k, query_vector=q_vec
        )
        t.retrieval_ms = int((time.monotonic() - r0) * 1000)
        top_score = retrieved[0].score if retrieved else None

        verdict = self._scope(question, q_vec, top_score)
        _, requested_choice = self._pick_generator(request.model)

        if verdict.declined:
            yield "token", DECLINE_MESSAGE
            meta = self._meta(
                generator=GeneratorKind.local_base,
                requested=requested_choice,
                audience=audience,
                retrieved_ids=[r.chunk_id for r in retrieved],
                timings=t,
                in_scope=False,
            )
            yield "meta", (meta, [f"declined: {verdict.reason}"], False)
            yield (
                "result",
                AskResponse(
                    answer_id=answer_id,
                    session_id=request.session_id,
                    prose=DECLINE_MESSAGE,
                    claims=[],
                    sources=[],
                    meta=meta,
                    declined=True,
                ),
            )
            return

        sources = [to_source_chunk(r) for r in retrieved]
        yield "sources", sources

        context_block = format_context(retrieved)
        generator, _ = self._pick_generator(request.model)
        deadline = time.monotonic() + self.settings.local_timeout_s

        g0 = time.monotonic()
        async for delta in generator.astream(question, context_block, audience, deadline=deadline):
            if delta.prose_delta:
                yield "token", delta.prose_delta
        answer = generator.last
        used_kind = answer.generator
        replaced = False

        if self._needs_fallback(answer) and c.hosted_generator is not None:
            hosted = await c.hosted_generator.collect(question, context_block, audience)
            if hosted.draft.prose.strip():
                answer = hosted
                used_kind = GeneratorKind.hosted_fallback
                replaced = True
                yield "token", ("__replace__", hosted.draft.prose)
        t.generation_ms = int((time.monotonic() - g0) * 1000)

        v0 = time.monotonic()
        claims: list[Claim] = verify_answer(answer.draft, retrieved, c.nli)  # type: ignore[arg-type]
        t.verification_ms = int((time.monotonic() - v0) * 1000)

        meta = self._meta(
            generator=used_kind,
            requested=requested_choice,
            audience=audience,
            retrieved_ids=[r.chunk_id for r in retrieved],
            timings=t,
            in_scope=True,
        )
        yield "claims", claims
        yield "meta", (meta, answer.warnings, replaced)
        yield (
            "result",
            AskResponse(
                answer_id=answer_id,
                session_id=request.session_id,
                prose=answer.draft.prose,
                claims=claims,
                sources=sources,
                meta=meta,
                declined=False,
            ),
        )

    # -- public -----------------------------------------------------

    async def answer_sync(self, request: AskRequest) -> AskResponse:
        result: AskResponse | None = None
        async for kind, payload in self._run(request):
            if kind == "result":
                result = payload
        assert result is not None
        return result

    async def answer_stream(self, request: AskRequest) -> AsyncIterator[SSEEvent]:
        async for kind, payload in self._run(request):
            if kind == "token":
                if isinstance(payload, tuple) and payload and payload[0] == "__replace__":
                    yield sse.token(payload[1], replace=True)
                else:
                    yield sse.token(payload)
            elif kind == "sources":
                yield sse.sources([s.model_dump(mode="json") for s in payload])
            elif kind == "claims":
                yield sse.claims([c.model_dump(mode="json") for c in payload])
            elif kind == "meta":
                meta, warnings, replaced = payload
                body = meta.model_dump(mode="json")
                body["warnings"] = warnings
                body["fell_back"] = replaced
                yield sse.meta(body)
            elif kind == "result":
                yield sse.done(payload.answer_id)
