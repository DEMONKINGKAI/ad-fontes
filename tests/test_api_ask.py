from __future__ import annotations


def test_ask_sync_503_when_components_not_loaded(client):
    # conftest sets eager_model_load=False, so no generator/NLI is loaded
    r = client.post("/api/ask/sync", json={"question": "What did Kai build?"})
    assert r.status_code == 503
    assert "loading" in r.json()["detail"].lower()


def test_ask_sync_length_cap(client):
    r = client.post("/api/ask/sync", json={"question": "x" * 401})
    assert r.status_code == 422
    assert "400" in r.json()["detail"]


def test_ask_rate_limit_kicks_in(client):
    # conftest sets rate_limit_requests=3
    body = {"question": "ok"}
    codes = [client.post("/api/ask/sync", json=body).status_code for _ in range(4)]
    assert codes[:3] == [503, 503, 503]  # allowed through to the pipeline
    assert codes[3] == 429


def test_ask_stream_emits_error_event_until_phase_2(client):
    with client.stream("POST", "/api/ask", json={"question": "hi"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert "event: error\n" in text
    assert "pipeline_not_ready" in text
    # each data line carries exactly one 'data: ' prefix (no double-encoding)
    assert "data: data:" not in text
    assert text.endswith("\n\n")


def test_ask_stream_is_rate_limited_separately(client):
    body = {"question": "ok"}
    for _ in range(3):
        client.post("/api/ask/sync", json=body)
    with client.stream("POST", "/api/ask", json=body) as r:
        text = "".join(r.iter_text())
    # 429 is raised before the stream body starts
    assert r.status_code == 429 or "rate limit" in text.lower()
