"""Shared generator interface + the incremental prose extractor.

Both backends (``local_llm``, ``hosted_llm``) emit the same JSON object
(``app.generation.schema.ANSWER_JSON_SCHEMA``): ``{"prose": ..., "claims": [...]}``.
The pipeline wants to *stream the prose* to the client while it is still being
generated, then parse the whole object for verification once generation ends
(brief §4). ``ProseStreamer`` does exactly that: fed the raw token stream, it
yields the delta of the ``prose`` string value and ignores everything else.

Keeping this here (rather than in ``local_llm``/``hosted_llm``) avoids a circular
import and lets it be unit-tested with no model.

(This module is a small addition to the brief's ``generation/`` file list — the
two backends genuinely need one shared contract.)
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from app.api.schemas import Audience, GeneratorKind
from app.generation.schema import AnswerDraft

_PROSE_OPEN = re.compile(r'"prose"\s*:\s*"')
_UNESCAPE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}


@dataclass(slots=True)
class GenerationDelta:
    """One streamed step. ``prose_delta`` is new user-visible text (may be empty)."""

    prose_delta: str = ""
    done: bool = False


@dataclass(slots=True)
class GeneratedAnswer:
    """Final parsed output of a generation run."""

    draft: AnswerDraft
    generator: GeneratorKind
    raw: str
    prose_streamed: str
    timed_out: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


class Generator(Protocol):
    """What the pipeline needs from a backend."""

    kind: GeneratorKind

    def astream(
        self, question: str, context_block: str, audience: Audience
    ) -> AsyncIterator[GenerationDelta]:
        """Yield prose deltas; the last item has ``done=True``."""
        ...

    async def collect(
        self, question: str, context_block: str, audience: Audience
    ) -> GeneratedAnswer:
        """Non-streaming convenience: run to completion and parse."""
        ...


# --------------------------------------------------------------------------- #
# Incremental prose extraction
# --------------------------------------------------------------------------- #


class ProseStreamer:
    """Feed it raw model output; get back the growing value of the ``prose`` key.

    The model is schema-constrained to emit ``{"prose": "<text>", "claims": [...]}``
    with ``prose`` first, so we scan for ``"prose"`` + ``:`` + the opening quote,
    then stream characters (handling JSON escapes) until the closing quote.
    """

    _SEEKING_KEY = 0
    _IN_VALUE = 1
    _DONE = 2

    def __init__(self) -> None:
        self._buf = ""
        self._state = self._SEEKING_KEY
        self._cursor = 0
        self._escaped = False
        self.prose = ""

    def feed(self, chunk: str) -> str:
        """Append raw output, return the new prose delta (possibly empty)."""
        self._buf += chunk
        if self._state == self._DONE:
            return ""
        if self._state == self._SEEKING_KEY:
            m = _PROSE_OPEN.search(self._buf)
            if not m:
                return ""
            self._state = self._IN_VALUE
            self._cursor = m.end()
            self._escaped = False

        delta_chars: list[str] = []
        i = self._cursor
        while i < len(self._buf):
            ch = self._buf[i]
            if self._escaped:
                delta_chars.append(_UNESCAPE.get(ch, ch))
                self._escaped = False
            elif ch == "\\":
                self._escaped = True
            elif ch == '"':
                self._state = self._DONE
                self._cursor = i + 1
                break
            else:
                delta_chars.append(ch)
            i += 1
        else:
            self._cursor = i

        delta = "".join(delta_chars)
        self.prose += delta
        return delta

    @property
    def raw(self) -> str:
        return self._buf


def parse_answer(raw: str) -> AnswerDraft:
    """Parse a (possibly trailing-garbage) model response into an AnswerDraft.

    llama.cpp grammar output is usually clean JSON, but hosted models sometimes
    wrap it in prose or a ```json fence — so we locate the outermost object.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("{") :] if "{" in text else text
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return AnswerDraft.model_validate(json.loads(text[start : i + 1]))
    # unterminated — try a best-effort close
    return AnswerDraft.model_validate(json.loads(text[start:] + '"}]}'))
