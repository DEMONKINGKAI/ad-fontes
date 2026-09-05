"""Hosted fallback generator via Hugging Face Inference Providers.

Used only when local generation exceeds ``local_timeout_s`` or errors. Requires
``HF_TOKEN`` (Kai's is HF Pro). The response is marked ``generator:
"hosted-fallback"`` so the UI can show it — the brief forbids failing silently to
a different model. If ``HF_TOKEN`` is unset the fallback is disabled and a local
failure surfaces as an error, never a silent swap.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.api.schemas import Audience, GeneratorKind
from app.generation.base import GeneratedAnswer, GenerationDelta, ProseStreamer, parse_answer
from app.generation.prompts import system_prompt, user_prompt
from app.generation.schema import AnswerDraft


class HostedGenerator:
    kind = GeneratorKind.hosted_fallback

    def __init__(
        self,
        model: str,
        token: str,
        *,
        provider: str = "auto",
        max_tokens: int = 640,
        temperature: float = 0.3,
    ) -> None:
        from huggingface_hub import InferenceClient

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = InferenceClient(
            model=model, token=token, provider=None if provider == "auto" else provider
        )

    def _messages(self, question: str, context_block: str, audience: Audience) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt(audience)},
            {"role": "user", "content": user_prompt(question, context_block)},
        ]

    def _blocking_complete(self, messages: list[dict]) -> str:
        resp = self._client.chat_completion(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    async def collect(
        self,
        question: str,
        context_block: str,
        audience: Audience,
        *,
        deadline: float | None = None,
    ) -> GeneratedAnswer:
        loop = asyncio.get_running_loop()
        messages = self._messages(question, context_block, audience)
        warnings: list[str] = []
        try:
            raw = await loop.run_in_executor(None, self._blocking_complete, messages)
        except Exception as exc:
            return GeneratedAnswer(
                draft=AnswerDraft(prose="(hosted fallback is unavailable)"),
                generator=self.kind,
                raw="",
                prose_streamed="",
                error=f"{type(exc).__name__}: {exc}",
                warnings=[f"hosted call failed: {exc}"],
            )
        streamer = ProseStreamer()
        streamer.feed(raw)
        try:
            draft = parse_answer(raw)
        except (ValueError, KeyError) as exc:
            warnings.append(f"parse failed: {exc}")
            draft = AnswerDraft(prose=streamer.prose.strip() or "(no readable answer)")
        return GeneratedAnswer(
            draft=draft,
            generator=self.kind,
            raw=raw,
            prose_streamed=streamer.prose,
            warnings=warnings,
        )

    async def astream(
        self,
        question: str,
        context_block: str,
        audience: Audience,
        *,
        deadline: float | None = None,
    ) -> AsyncIterator[GenerationDelta]:
        answer = await self.collect(question, context_block, audience)
        self.last = answer
        if answer.draft.prose:
            yield GenerationDelta(prose_delta=answer.draft.prose)
        yield GenerationDelta(done=True)
