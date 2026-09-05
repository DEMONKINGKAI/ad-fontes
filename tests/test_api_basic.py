from __future__ import annotations

import json


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "ad-fontes"


def test_liveness(client):
    assert client.get("/api/health/live").json() == {"status": "alive"}


def test_health_reports_starting_before_models_load(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "starting"
    assert body["corpus_version"] == "2026-09-05"
    assert body["index_size"] == 0
    assert body["default_model"] == "tuned"
    assert body["hosted_fallback_configured"] is False
    names = {m["name"] for m in body["models"]}
    assert {"retriever", "nli", "generator:base", "generator:tuned"} <= names


def test_openapi_served(client):
    assert client.get("/openapi.json").status_code == 200


def test_projects_lists_all_seven(client):
    r = client.get("/api/projects")
    assert r.status_code == 200
    rows = r.json()
    ids = {row["id"] for row in rows}
    assert ids == {
        "threadfall",
        "causeway",
        "fons-iuris",
        "evidentia",
        "pharmacausal",
        "neumf",
        "loan-approval",
    }
    fons = next(row for row in rows if row["id"] == "fons-iuris")
    assert fons["repo"] == "https://github.com/DEMONKINGKAI/fons-iuris"
    assert "RAG" in fons["summary"] or "claim" in fons["summary"].lower()
    loan = next(row for row in rows if row["id"] == "loan-approval")
    assert loan["repo"] is None  # manifest says "null (private code; ...)"


def test_feedback_appends_jsonl(client, settings):
    payload = {
        "session_id": "s1",
        "question": "What did Kai build?",
        "answer_id": "a-123",
        "rating": "up",
        "note": "clear",
    }
    r = client.post("/api/feedback", json=payload)
    assert r.status_code == 200
    assert r.json() == {"stored": True, "answer_id": "a-123"}
    files = list(settings.feedback_dir.glob("feedback-*.jsonl"))
    assert len(files) == 1
    row = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert row["rating"] == "up"
    assert row["answer_id"] == "a-123"
    assert "ip" not in row


def test_feedback_rejects_bad_rating(client):
    r = client.post(
        "/api/feedback",
        json={"session_id": "s", "question": "q", "answer_id": "a", "rating": "maybe"},
    )
    assert r.status_code == 422
