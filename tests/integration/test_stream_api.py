"""POST /session, GET /session/{id}, WS /session/{id}/stream — end to end against an in-memory
Qdrant + the synthetic fixture corpora (docs/API.md, REQ-STREAM-*, REQ-SEC-06).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from starlette.websockets import WebSocketDisconnect

from app.controller.entity_extraction import get_corpus_matcher
from app.core.config import get_settings
from app.core.slots import get_slot_schema
from scripts.ingest_corpus import ingest_corpus
from tests.fakes import FIXTURE_CORPORA, FakeEmbedder

API_KEY = "change_me_local_dev"


@pytest.fixture
async def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_ROOT", str(FIXTURE_CORPORA))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("API_KEY", API_KEY)
    get_settings.cache_clear()
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()

    settings = get_settings()
    embedder = FakeEmbedder()
    qdrant_client = AsyncQdrantClient(location=":memory:")
    for corpus_id in ("alpha", "beta"):
        await ingest_corpus(
            corpus_id, settings=settings, embedder=embedder, qdrant_client=qdrant_client
        )

    import app.main as main_module

    monkeypatch.setattr(main_module, "create_qdrant_client", lambda url: qdrant_client)
    monkeypatch.setattr(main_module, "get_embedder", lambda: embedder)

    app = main_module.create_app()
    with TestClient(app) as client:
        yield client

    get_settings.cache_clear()
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


def auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def create_session(client: TestClient, corpus_id: str = "alpha") -> dict:
    resp = client.post("/session", json={"corpus_id": corpus_id}, headers=auth_header())
    assert resp.status_code == 201, resp.text
    return resp.json()


def send_chunk(ws, seq: int, text: str, t_offset_ms: int, is_final: bool = False) -> None:
    ws.send_json(
        {
            "event_type": "TRANSCRIPT_CHUNK",
            "payload": {
                "seq": seq,
                "text_delta": text,
                "t_offset_ms": t_offset_ms,
                "is_final": is_final,
            },
        }
    )


def receive_n(ws, count: int) -> list[dict]:
    return [ws.receive_json() for _ in range(count)]


def receive_retrieve_turn(ws) -> list[dict]:
    """One RETRIEVE turn emits, in this fixed order: RETRIEVAL_DECISION (emitted before the
    retriever call), then RETRIEVAL_STARTED x2 and RETRIEVAL_COMPLETED x2 (dense+sparse, from
    the Phase 2 HybridRetriever's own telemetry)."""
    events = receive_n(ws, 5)
    assert events[0]["event_type"] == "RETRIEVAL_DECISION"
    assert events[0]["payload"]["decision"] == "RETRIEVE"
    types = [e["event_type"] for e in events[1:]]
    assert sorted(types) == [
        "RETRIEVAL_COMPLETED",
        "RETRIEVAL_COMPLETED",
        "RETRIEVAL_STARTED",
        "RETRIEVAL_STARTED",
    ]
    return events


# --- POST /session, GET /session/{id} -----------------------------------------------------------


def test_create_session_returns_session_id_and_ws_url(app_client):
    body = create_session(app_client)
    assert body["session_id"]
    assert body["ws_url"] == f"/session/{body['session_id']}/stream?token={API_KEY}"
    assert "created_at" in body


def test_create_session_unknown_corpus_returns_400(app_client):
    resp = app_client.post("/session", json={"corpus_id": "gamma"}, headers=auth_header())
    assert resp.status_code == 400


def test_create_session_requires_api_key(app_client):
    resp = app_client.post("/session", json={"corpus_id": "alpha"})
    assert resp.status_code == 401


def test_get_session_returns_snapshot(app_client):
    created = create_session(app_client)
    resp = app_client.get(f"/session/{created['session_id']}", headers=auth_header())
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == created["session_id"]
    assert body["corpus_id"] == "alpha"
    assert body["status"] == "active"
    assert body["entities"] == {}
    assert body["latest_answer_version"] is None


def test_get_unknown_session_returns_404(app_client):
    resp = app_client.get("/session/sess_does_not_exist", headers=auth_header())
    assert resp.status_code == 404


# --- WS auth (REQ-SEC-06) ------------------------------------------------------------------------


def test_ws_requires_valid_token(app_client):
    """docs/API.md §3: closes with code 4401 on auth failure. The server accepts first, then
    closes with the specific code — a pre-accept close can only surface as a generic HTTP
    rejection to a real client, not this application-level code (verified against a live
    Docker stack with a real WS client during Phase 3's manual E2E check)."""
    session_id = create_session(app_client)["session_id"]
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        app_client.websocket_connect(f"/session/{session_id}/stream?token=wrong") as ws,
    ):
        ws.receive_json()  # the close already happened server-side; surfaces here
    assert exc_info.value.code == 4401


def test_ws_unknown_session_closes_4404(app_client):
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        app_client.websocket_connect(f"/session/sess_nope/stream?token={API_KEY}") as ws,
    ):
        ws.receive_json()
    assert exc_info.value.code == 4404


# --- WS streaming / controller path (docs/API.md §3) --------------------------------------------


BASE_UTTERANCE = (
    "I am planning a corporate event next month and I need to book a venue "
    "that can comfortably fit everyone"
)


def test_ws_streaming_produces_wait_then_retrieve(app_client):
    """The first chunk can't be judged stable (needs >=2 embeddings) and has no entity yet ->
    WAIT. A near-duplicate second chunk that appends the entity is both stable (high word overlap,
    same reasoning as test_retrieve_end_to_end_without_mocking_stability) and actionable ->
    RETRIEVE, with the Phase 2 dense+sparse telemetry riding the same socket."""
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, BASE_UTTERANCE, 0)
        decision = ws.receive_json()
        assert decision["event_type"] == "RETRIEVAL_DECISION"
        assert decision["payload"]["decision"] == "WAIT"
        assert decision["session_id"] == session_id

        send_chunk(ws, 1, BASE_UTTERANCE + ", for 30 guests", 500)
        receive_retrieve_turn(ws)


def test_ws_no_retrieval_call_when_decision_is_wait_or_no_retrieval(app_client):
    """No RETRIEVAL_STARTED/COMPLETED must ever appear around a WAIT decision. Proven by sending
    two WAIT-only chunks back to back and checking each produces exactly its own single event —
    a leaked retrieval call would show up as an extra event received out of turn."""
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "um so anyway", 0)
        first = ws.receive_json()
        assert first["event_type"] == "RETRIEVAL_DECISION"
        assert first["payload"]["decision"] == "WAIT"

        send_chunk(ws, 1, "completely unrelated words here", 400)
        second = ws.receive_json()
        assert second["event_type"] == "RETRIEVAL_DECISION"
        assert second["payload"]["decision"] == "WAIT"
        assert second["trace_id"] != first["trace_id"]


def test_ws_invalid_frame_emits_error_and_keeps_the_connection_open(app_client):
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        ws.send_json({"event_type": "NOT_A_CHUNK", "payload": {}})
        error = ws.receive_json()
        assert error["event_type"] == "ERROR"
        assert error["payload"]["stage"] == "stream"

        send_chunk(ws, 0, "hello", 0)  # connection still usable afterward
        decision = ws.receive_json()
        assert decision["event_type"] == "RETRIEVAL_DECISION"


def test_ws_trace_id_is_shared_within_one_chunk_but_differs_across_chunks(app_client):
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, BASE_UTTERANCE, 0)
        first = ws.receive_json()
        assert first["payload"]["decision"] == "WAIT"  # only 1 embedding so far, can't be stable

        send_chunk(ws, 1, BASE_UTTERANCE + ", for 30 guests", 500)
        turn_2 = receive_retrieve_turn(ws)  # RETRIEVE -> one trace_id across all 5 of its events
        assert len({e["trace_id"] for e in turn_2}) == 1
        assert turn_2[0]["trace_id"] != first["trace_id"]


# --- corpus isolation over the API ----------------------------------------------------------------


def test_two_sessions_on_different_corpora_retrieve_independently(app_client):
    alpha_session = create_session(app_client, "alpha")["session_id"]
    beta_session = create_session(app_client, "beta")["session_id"]

    with app_client.websocket_connect(f"/session/{alpha_session}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, BASE_UTTERANCE, 0)
        ws.receive_json()
        send_chunk(ws, 1, BASE_UTTERANCE + ", for 30 guests", 500)
        turn = receive_retrieve_turn(ws)
        completed = [e for e in turn if e["event_type"] == "RETRIEVAL_COMPLETED"]
        assert any(c["payload"]["result_count"] > 0 for c in completed)

    with app_client.websocket_connect(f"/session/{beta_session}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, BASE_UTTERANCE, 0)
        ws.receive_json()
        send_chunk(ws, 1, BASE_UTTERANCE + ", for 30 guests", 500)
        # beta has no "guests"/venue vocabulary at all -> still no entity ever found -> WAIT.
        decision = ws.receive_json()
        assert decision["payload"]["decision"] == "WAIT"


# --- SESSION_RESYNC (REQ-STREAM-03) ---------------------------------------------------------------


def test_first_connection_does_not_emit_session_resync(app_client):
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "hello", 0)
        first_event = ws.receive_json()
        assert first_event["event_type"] != "SESSION_RESYNC"


def test_reconnect_resumes_state(app_client):
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "Orion Hall for 30 guests", 0)
        ws.receive_json()

    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        resync = ws.receive_json()
        assert resync["event_type"] == "SESSION_RESYNC"
        assert resync["session_id"] == session_id
        assert resync["payload"]["entities"] == {"venue": "Orion Hall", "capacity": "30"}
        assert resync["payload"]["last_seq"] == 0
        assert resync["payload"]["latest_answer_version"] is None

        # the client does not replay prior chunks — only new ones are sent, continuing the seq.
        send_chunk(ws, 1, ", and the cancellation policy", 100)
        decision = ws.receive_json()
        assert decision["event_type"] == "RETRIEVAL_DECISION"
