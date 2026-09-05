"""Server-Sent Events helpers for ``POST /api/ask``.

The frontend consumes this stream with a plain ``EventSource``/``fetch`` reader,
so the wire format matters: each event is ``event: <name>\\n`` followed by one or
more ``data: <json>\\n`` lines and a blank line. Payloads are JSON so the client
does not have to parse anything bespoke.

Event sequence for a normal answer:
    token*   incremental prose deltas
    sources  the retrieved chunk set (once, after retrieval)
    claims   the verified structured claims (once, after generation+verification)
    meta     timings and generator kind (once)
    done     terminal marker
On failure a single ``error`` event replaces ``claims``/``meta``/``done``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

SSE_EVENTS = ("token", "sources", "claims", "meta", "done", "error")


@dataclass(slots=True)
class SSEEvent:
    event: str
    data: Any

    def encode(self) -> str:
        if self.event not in SSE_EVENTS:
            raise ValueError(f"unknown SSE event: {self.event!r}")
        payload = (
            self.data
            if isinstance(self.data, str)
            else json.dumps(self.data, default=_json_default)
        )
        lines = [f"event: {self.event}"]
        lines.extend(f"data: {chunk}" for chunk in payload.split("\n"))
        return "\n".join(lines) + "\n\n"


def _json_default(obj: object) -> object:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")


def token(text: str) -> SSEEvent:
    return SSEEvent("token", {"text": text})


def sources(payload: Any) -> SSEEvent:
    return SSEEvent("sources", payload)


def claims(payload: Any) -> SSEEvent:
    return SSEEvent("claims", payload)


def meta(payload: Any) -> SSEEvent:
    return SSEEvent("meta", payload)


def done(answer_id: str) -> SSEEvent:
    return SSEEvent("done", {"answer_id": answer_id})


def error(message: str, detail: str | None = None) -> SSEEvent:
    return SSEEvent("error", {"error": message, "detail": detail})


def encode_stream(events: Iterable[SSEEvent]) -> Iterable[str]:
    for ev in events:
        yield ev.encode()


async def encode_astream(events: AsyncIterator[SSEEvent]) -> AsyncIterator[str]:
    async for ev in events:
        yield ev.encode()
