"""llama-cpp-python backend for the ``base`` and ``tuned`` GGUF generators.

Both models are Q4_K_M GGUF loaded once at startup (brief §2). Output is
constrained to valid JSON (``response_format`` — see ``grammar_mode`` on the
class and ARCHITECTURE.md for why a full JSON-schema grammar is not the default)
with the ``{"prose", "claims"}`` shape described in the prompt. The prose is
streamed as it is generated (``ProseStreamer``); ``claims`` is parsed once
generation finishes.

llama.cpp has no clean mid-generation stop for chat completions, so generation
runs on a worker thread and the async side abandons it once ``deadline`` passes;
``max_tokens`` bounds a runaway regardless. The pipeline uses that to fall back
to the hosted model.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from app.api.schemas import Audience, GeneratorKind
from app.generation.base import GeneratedAnswer, GenerationDelta, ProseStreamer, parse_answer
from app.generation.prompts import system_prompt, user_prompt
from app.generation.schema import ANSWER_JSON_SCHEMA, AnswerDraft

_SENTINEL = object()


@dataclass(slots=True)
class _Err:
    exc: BaseException


class LocalGenerator:
    def __init__(
        self,
        kind: GeneratorKind,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        grammar_mode: str = "json_object",
    ) -> None:
        from llama_cpp import Llama

        self.kind = kind
        self.max_tokens = max_tokens
        self.temperature = temperature
        # grammar_mode:
        #   "json_object" — constrain to *any* valid JSON (the schema lives in the
        #                   prompt). Default: full JSON-schema grammars crash the
        #                   current llama-cpp-python build (stack overflow in the
        #                   grammar sampler). See ARCHITECTURE.md.
        #   "schema"      — try the JSON-schema grammar (may work on other builds).
        #   "none"        — no constraint; rely on the prompt + robust parsing.
        self.grammar_mode = grammar_mode
        self._response_format: dict | None
        if grammar_mode == "json_object":
            self._response_format = {"type": "json_object"}
        elif grammar_mode == "schema":
            self._response_format = {"type": "json_object", "schema": ANSWER_JSON_SCHEMA}
        else:
            self._response_format = None
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=False,
            verbose=False,
        )
        self._lock = asyncio.Lock()

    # -- internals --------------------------------------------------------

    def _messages(self, question: str, context_block: str, audience: Audience) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt(audience)},
            {"role": "user", "content": user_prompt(question, context_block)},
        ]

    def _run_stream(self, messages: list[dict]):
        return self._llm.create_chat_completion(
            messages=messages,
            response_format=self._response_format,
            stream=True,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _worker(self, messages, abort, put) -> None:
        try:
            for item in self._run_stream(messages):
                if abort.is_set():
                    break
                put(item)
        except BaseException as exc:
            put(_Err(exc))
        finally:
            put(_SENTINEL)

    # -- public API -----------------------------------------------------

    async def astream(
        self,
        question: str,
        context_block: str,
        audience: Audience,
        *,
        deadline: float | None = None,
    ) -> AsyncIterator[GenerationDelta]:
        """Yield prose deltas. The pipeline reads ``self.last`` for the full result.

        llama.cpp has no clean mid-generation stop for chat completions, so
        generation runs on a worker thread and this coroutine abandons it when
        ``deadline`` passes (``abort`` stops it within one token; ``max_tokens``
        bounds it regardless).
        """
        loop = asyncio.get_running_loop()
        streamer = ProseStreamer()
        messages = self._messages(question, context_block, audience)
        queue: asyncio.Queue = asyncio.Queue()
        abort = threading.Event()
        timed_out = False
        err: BaseException | None = None

        def put(item: object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        async with self._lock:  # one Llama instance is not concurrency-safe
            worker = loop.run_in_executor(None, self._worker, messages, abort, put)
            try:
                while True:
                    timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout)
                    except TimeoutError:
                        timed_out = True
                        abort.set()
                        break
                    if item is _SENTINEL:
                        break
                    if isinstance(item, _Err):
                        err = item.exc
                        break
                    piece = item["choices"][0].get("delta", {}).get("content") or ""
                    delta = streamer.feed(piece) if piece else ""
                    if delta:
                        yield GenerationDelta(prose_delta=delta)
            finally:
                abort.set()
                if not timed_out:
                    await worker  # normal path: thread already finished

            self.last = self._finalise(streamer, timed_out, err)
        yield GenerationDelta(done=True, answer=self.last)

    def _finalise(
        self, streamer: ProseStreamer, timed_out: bool, err: BaseException | None = None
    ) -> GeneratedAnswer:
        raw = streamer.raw
        warnings: list[str] = []
        if err is not None:
            warnings.append(f"generation error: {type(err).__name__}: {err}")
        try:
            draft = parse_answer(raw)
        except (ValueError, KeyError) as exc:
            warnings.append(f"parse failed: {exc}")
            fallback = streamer.prose.strip() or "(generation produced no readable answer)"
            draft = AnswerDraft(prose=fallback)
        return GeneratedAnswer(
            draft=draft,
            generator=self.kind,
            raw=raw,
            prose_streamed=streamer.prose,
            timed_out=timed_out,
            error=(f"{type(err).__name__}: {err}" if err else None),
            warnings=warnings,
        )

    async def collect(
        self,
        question: str,
        context_block: str,
        audience: Audience,
        *,
        deadline: float | None = None,
    ) -> GeneratedAnswer:
        answer: GeneratedAnswer | None = None
        async for delta in self.astream(question, context_block, audience, deadline=deadline):
            if delta.answer is not None:
                answer = delta.answer
        assert answer is not None
        return answer


def load_local_generator(
    kind: GeneratorKind,
    repo: str,
    filename: str,
    *,
    cache_dir: str | Path | None = None,
    **kwargs,
) -> LocalGenerator:
    """Download the GGUF (cached) and construct the generator."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo, filename=filename, cache_dir=str(cache_dir) if cache_dir else None
    )
    return LocalGenerator(kind, path, **kwargs)
