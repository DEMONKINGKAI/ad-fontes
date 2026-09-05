from __future__ import annotations

import json

import pytest
from app.api import sse


def test_token_event_encoding():
    out = sse.token("hello").encode()
    assert out.startswith("event: token\n")
    assert out.endswith("\n\n")
    body = out.split("data: ", 1)[1].strip()
    assert json.loads(body) == {"text": "hello"}


def test_multiline_data_is_split_into_data_lines():
    out = sse.SSEEvent("token", "a\nb").encode()
    assert "data: a\ndata: b\n\n" in out


def test_unknown_event_rejected():
    with pytest.raises(ValueError, match="unknown SSE event"):
        sse.SSEEvent("bogus", {}).encode()


def test_error_event_shape():
    payload = json.loads(sse.error("boom", "detail").encode().split("data: ", 1)[1].strip())
    assert payload == {"error": "boom", "detail": "detail"}
