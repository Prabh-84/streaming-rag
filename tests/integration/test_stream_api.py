"""POST /session, GET /session/{id}, WS /session/{id}/stream — end to end against an in-memory
Qdrant + the synthetic fixture corpora (docs/API.md, REQ-STREAM-*, REQ-SEC-06).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from starlette.websockets import WebSocketDisconnect

import app.controller.retrieval_controller as retrieval_controller
import app.decomposition.multi_intent as multi_intent
import app.generation.streaming_generator as streaming_generator
from app.controller.entity_extraction import get_corpus_matcher
from app.core.config import get_settings
from app.core.slots import get_slot_schema
from scripts.ingest_corpus import ingest_corpus
from tests.fakes import (
    FIXTURE_CORPORA,
    FakeDecomposer,
    FakeEmbedder,
    FakeGenerationLLM,
    FakeReranker,
)

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
    # Safe default: no usable sub-queries, so any incidentally-compound test text still falls
    # back to one sub-query rather than ever reaching a real Anthropic call. Individual tests
    # that want genuine decomposition re-patch this within their own scope.
    monkeypatch.setattr(multi_intent, "get_decomposer", lambda: FakeDecomposer(sub_queries=None))
    # Same reasoning for reranking (Phase 5): never let a WS integration test load the real
    # cross-encoder model just because retrieval against the real ingested corpus returns
    # non-empty results.
    monkeypatch.setattr(retrieval_controller, "get_reranker", lambda: FakeReranker())
    # Same reasoning for generation (Phase 7): an empty-stream FakeGenerationLLM produces no
    # sentences, so stream_answer() returns None and emits nothing - existing tests' event counts
    # are unaffected unless a test explicitly wants to exercise generation.
    monkeypatch.setattr(streaming_generator, "get_generator", lambda: FakeGenerationLLM())

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


def receive_retrieve_turn(ws, *, expected_sub_queries: int = 1) -> list[dict]:
    """One RETRIEVE turn emits, in this fixed order: RETRIEVAL_DECISION (emitted before
    decomposition), one SUBQUERY_CREATED per sub-query (Phase 4 — exactly 1 for a non-compound
    request), RETRIEVAL_STARTED/RETRIEVAL_COMPLETED per sub-query per mode (dense+sparse, Phase 2
    HybridRetriever telemetry), then one RERANK_COMPLETED per sub-query (Phase 5, emitted only
    after every sub-query's retrieval has finished — fusion needs the whole turn's results)."""
    total = 1 + expected_sub_queries + 4 * expected_sub_queries + expected_sub_queries
    events = receive_n(ws, total)
    assert events[0]["event_type"] == "RETRIEVAL_DECISION"
    assert events[0]["payload"]["decision"] == "RETRIEVE"
    subquery_events = events[1 : 1 + expected_sub_queries]
    assert all(e["event_type"] == "SUBQUERY_CREATED" for e in subquery_events)
    retrieval_events = events[1 + expected_sub_queries : 1 + 5 * expected_sub_queries]
    retrieval_types = [e["event_type"] for e in retrieval_events]
    assert sorted(retrieval_types) == sorted(
        ["RETRIEVAL_STARTED", "RETRIEVAL_STARTED", "RETRIEVAL_COMPLETED", "RETRIEVAL_COMPLETED"]
        * expected_sub_queries
    )
    rerank_events = events[1 + 5 * expected_sub_queries :]
    assert all(e["event_type"] == "RERANK_COMPLETED" for e in rerank_events)
    assert len(rerank_events) == expected_sub_queries
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


def test_ws_grounded_answer_streams_citations_and_answer_version(app_client, monkeypatch):
    """Phase 7 end to end over the real WebSocket path: a RETRIEVE turn against the real ingested
    alpha corpus produces a cited sentence, CITATION_CREATED and ANSWER_DELTA events, and a final
    ANSWER_VERSION_CREATED — using the real Cancellation Policy content, never a benchmark string.
    orion_hall.md's third heading ("Cancellation Policy") is real section label "§3" (scripts/
    ingest_corpus.py's split_sections numbers headings in document order, 1-based) — verified
    directly against split_sections() output, not assumed from the heading text."""
    monkeypatch.setattr(
        streaming_generator,
        "get_generator",
        lambda: FakeGenerationLLM(
            chunks=[
                "A booking at Orion Hall can be cancelled free of charge up to 14 days before the "
                "event [orion_hall §3]."
            ]
        ),
    )
    session_id = create_session(app_client)["session_id"]
    base = "I need to know more about Orion Hall for my upcoming event booking"
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, base, 0)
        decision = ws.receive_json()
        assert decision["payload"]["decision"] == "WAIT"  # first turn: no generation triggered

        # High word overlap with chunk 0 keeps stability high (real bge-small model, same
        # reasoning as test_ws_streaming_produces_wait_then_retrieve) while the appended clause
        # supplies the cancellation_policy entity; verified empirically to also clear
        # MIN_RELEVANCE under FakeReranker's word-overlap scoring for the Cancellation Policy
        # chunk specifically, which the fake generator's scripted citation targets.
        send_chunk(ws, 1, base + ", the cancellation policy", 500)
        receive_retrieve_turn(ws)

        citation = ws.receive_json()
        assert citation["event_type"] == "CITATION_CREATED"
        assert citation["payload"]["doc_id"] == "orion_hall"
        assert citation["payload"]["section"] == "§3"

        answer_delta = ws.receive_json()
        assert answer_delta["event_type"] == "ANSWER_DELTA"
        assert answer_delta["payload"]["is_final_sentence"] is True
        assert "[orion_hall §3]" in answer_delta["payload"]["text_delta"]

        answer_version = ws.receive_json()
        assert answer_version["event_type"] == "ANSWER_VERSION_CREATED"
        assert answer_version["payload"]["version_no"] == 1
        assert answer_version["payload"]["supersedes"] is None
        assert len(answer_version["payload"]["citations"]) == 1


def test_compound_ws_request_produces_multiple_subqueries(app_client, monkeypatch):
    """Phase 4 end to end over the real WebSocket path: a genuinely compound utterance decomposes
    into 2 SUBQUERY_CREATED events and 2 concurrently-retrieved sub-queries, using the real corpus
    content (alpha's Cancellation Policy / Catering sections) — never a benchmark string."""
    monkeypatch.setattr(
        multi_intent,
        "get_decomposer",
        lambda: FakeDecomposer(
            sub_queries=[
                {
                    "text": "What is the cancellation policy for Orion Hall?",
                    "intent_label": "cancellation",
                },
                {
                    "text": "What are the catering options for Orion Hall?",
                    "intent_label": "catering",
                },
            ]
        ),
    )
    session_id = create_session(app_client)["session_id"]
    base = (
        "I am planning a corporate event next month and I need to book a venue that can "
        "comfortably fit everyone and I have several detailed questions about the arrangements "
        "before I can confirm the final booking with my team"
    )
    compound_clause = ", the cancellation policy and the catering options for Orion Hall"
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, base, 0)
        decision = ws.receive_json()
        assert decision["payload"]["decision"] == "WAIT"  # first chunk: no entity yet

        # High word overlap with chunk 0 (same reasoning as test_ws_streaming_produces_wait_then_
        # retrieve) keeps stability high while the appended clause supplies both new entities and
        # the compound (cc/conj across two distinct slots) signal. This restates chunk 0's text in
        # chunk 1's delta rather than sending only the incremental clause, which duplicates it in
        # session.buffer_text() (the per-chunk stability embedding, keyed on text_delta alone,
        # would otherwise see too little word overlap between chunks with the hash-based test
        # embedder). Harmless here since the mocked decomposer returns fixed sub-query text rather
        # than echoing the buffer; see the Phase 4 report for this as a pre-existing test-fixture
        # characteristic inherited from Phase 3, not an application bug.
        send_chunk(ws, 1, base + compound_clause, 500)
        turn = receive_retrieve_turn(ws, expected_sub_queries=2)

        subquery_events = [e for e in turn if e["event_type"] == "SUBQUERY_CREATED"]
        assert {e["payload"]["intent_label"] for e in subquery_events} == {
            "cancellation",
            "catering",
        }

        started = [e for e in turn if e["event_type"] == "RETRIEVAL_STARTED"]
        assert {e["payload"]["trigger"] for e in started} == {"provisional", "multi_intent"}


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
        turn_2 = receive_retrieve_turn(ws)  # RETRIEVE -> one trace_id across all 7 of its events
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


# --- Phase 8: Telemetry / Observability (REQ-OBS-01..04) -----------------------------------------


def get_events_json(client: TestClient, session_id: str, **params) -> list[dict]:
    query = "&".join(["format=json", *(f"{k}={v}" for k, v in params.items())])
    resp = client.get(f"/session/{session_id}/events?{query}", headers=auth_header())
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_events_endpoint_requires_api_key(app_client):
    session_id = create_session(app_client)["session_id"]
    resp = app_client.get(f"/session/{session_id}/events?format=json")
    assert resp.status_code == 401


def test_events_endpoint_unknown_session_returns_404(app_client):
    resp = app_client.get("/session/nonexistent/events?format=json", headers=auth_header())
    assert resp.status_code == 404


def test_events_endpoint_default_format_is_sse(app_client):
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "hello", 0)
        ws.receive_json()

    resp = app_client.get(f"/session/{session_id}/events", headers=auth_header())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.startswith("data: ")


# --- 1. Every required stage transition emits telemetry / 2. payload/schema correctness ----------


def test_persisted_events_cover_the_full_retrieve_turn_with_correct_schemas(app_client):
    """REQ-OBS-01/02: every event a RETRIEVE turn produces over the live socket is also
    independently queryable from the persistent store afterwards, with the exact documented
    payload fields (docs/TELEMETRY.md §2)."""
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, BASE_UTTERANCE, 0)
        ws.receive_json()
        send_chunk(ws, 1, BASE_UTTERANCE + ", for 30 guests", 500)
        receive_retrieve_turn(ws)

    events = get_events_json(app_client, session_id)
    by_type = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)

    decision = by_type["RETRIEVAL_DECISION"][-1]
    assert {"decision", "trigger", "reason"} <= decision["payload"].keys()
    assert decision["payload"]["decision"] == "RETRIEVE"

    subquery = by_type["SUBQUERY_CREATED"][0]
    assert {"sub_query_id", "text", "intent_label"} <= subquery["payload"].keys()

    started = by_type["RETRIEVAL_STARTED"][0]
    assert {"sub_query_id", "mode", "t_offset_ms", "trigger"} <= started["payload"].keys()

    completed = by_type["RETRIEVAL_COMPLETED"][0]
    assert {
        "sub_query_id",
        "mode",
        "result_count",
        "latency_ms",
        "result_chunk_ids",
        "scores",
        "t_offset_ms",
        "trigger",
    } <= completed["payload"].keys()

    rerank = by_type["RERANK_COMPLETED"][0]
    assert {"sub_query_id", "ranked_chunk_ids", "scores"} <= rerank["payload"].keys()

    # every event this turn produced shares one trace_id (REQ-OBS-01's envelope contract).
    turn_trace_ids = {e["trace_id"] for e in events if e["event_type"] != "RETRIEVAL_DECISION"} | {
        decision["trace_id"]
    }
    assert len(turn_trace_ids) == 1
    assert all(e["session_id"] == session_id for e in events)


# --- 3. SESSION_RESYNC telemetry ---------------------------------------------------------------


def test_session_resync_is_persisted_and_queryable(app_client):
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "Orion Hall for 30 guests", 0)
        ws.receive_json()

    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        ws.receive_json()  # the live SESSION_RESYNC frame itself

    events = get_events_json(app_client, session_id, event_type="SESSION_RESYNC")
    assert len(events) == 1
    assert events[0]["payload"]["entities"] == {"venue": "Orion Hall", "capacity": "30"}


# --- 6. Telemetry under multi-intent retrieval ------------------------------------------------


def test_multi_intent_events_are_all_persisted(app_client, monkeypatch):
    monkeypatch.setattr(
        multi_intent,
        "get_decomposer",
        lambda: FakeDecomposer(
            sub_queries=[
                {"text": "cancellation policy for Orion Hall", "intent_label": "cancellation"},
                {"text": "catering options for Orion Hall", "intent_label": "catering"},
            ]
        ),
    )
    session_id = create_session(app_client)["session_id"]
    base = (
        "I am planning a corporate event next month and I need to book a venue that can "
        "comfortably fit everyone and I have several detailed questions about the arrangements "
        "before I can confirm the final booking with my team"
    )
    compound_clause = ", the cancellation policy and the catering options for Orion Hall"
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, base, 0)
        ws.receive_json()
        send_chunk(ws, 1, base + compound_clause, 500)
        receive_retrieve_turn(ws, expected_sub_queries=2)

    events = get_events_json(app_client, session_id)
    subqueries = [e for e in events if e["event_type"] == "SUBQUERY_CREATED"]
    started = [e for e in events if e["event_type"] == "RETRIEVAL_STARTED"]
    rerank = [e for e in events if e["event_type"] == "RERANK_COMPLETED"]
    assert len(subqueries) == 2
    assert len(started) == 4  # 2 sub-queries x (dense, sparse)
    assert len(rerank) == 2
    assert {e["payload"]["trigger"] for e in started} == {"provisional", "multi_intent"}


# --- 7. Telemetry during session refinement --------------------------------------------------


def test_refinement_events_are_persisted(app_client, monkeypatch):
    """A second RETRIEVE turn (REQ-SESS-01 refinement, Phase 6) still lands its own
    RETRIEVAL_DECISION/SUBQUERY_CREATED/etc in the persistent store, tagged with the delta
    trigger."""
    monkeypatch.setattr(retrieval_controller, "classify_segment", lambda *a, **k: "refinement")
    session_id = create_session(app_client)["session_id"]
    base = "I need to know more about Orion Hall for my upcoming event booking"
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, base, 0)
        ws.receive_json()
        send_chunk(ws, 1, base + ", for 30 guests", 500)
        receive_retrieve_turn(ws)
        ws.receive_json()  # UNCERTAINTY - grounded generation finds this vague query's evidence
        # insufficient (Phase 7); every RETRIEVE turn runs stream_answer regardless of trigger,
        # so this trailing event must be drained before the next chunk's own decision arrives.

        send_chunk(ws, 2, ", cancellation policy", 1000)
        second_decision = ws.receive_json()
        assert second_decision["payload"]["trigger"] == "delta"
        receive_n(ws, 6)  # subquery_created + 4 retrieval + rerank_completed for the delta turn

    events = get_events_json(app_client, session_id)
    decisions = [e for e in events if e["event_type"] == "RETRIEVAL_DECISION"]
    # chunk 0's own WAIT decision (trigger=None) is persisted too - only the two RETRIEVE
    # decisions that follow it carry a trigger.
    assert [d["payload"]["trigger"] for d in decisions] == [None, "provisional", "delta"]


# --- 8. Telemetry during grounded answer generation --------------------------------------------


def test_generation_events_are_persisted(app_client, monkeypatch):
    monkeypatch.setattr(
        streaming_generator,
        "get_generator",
        lambda: FakeGenerationLLM(
            chunks=[
                "A booking at Orion Hall can be cancelled free of charge up to 14 days before the "
                "event [orion_hall §3]."
            ]
        ),
    )
    session_id = create_session(app_client)["session_id"]
    base = "I need to know more about Orion Hall for my upcoming event booking"
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, base, 0)
        ws.receive_json()
        send_chunk(ws, 1, base + ", the cancellation policy", 500)
        receive_retrieve_turn(ws)
        ws.receive_json()  # CITATION_CREATED
        ws.receive_json()  # ANSWER_DELTA
        ws.receive_json()  # ANSWER_VERSION_CREATED

    events = get_events_json(app_client, session_id)
    assert any(e["event_type"] == "CITATION_CREATED" for e in events)
    assert any(e["event_type"] == "ANSWER_DELTA" for e in events)
    version_events = [e for e in events if e["event_type"] == "ANSWER_VERSION_CREATED"]
    assert len(version_events) == 1
    assert version_events[0]["payload"]["version_no"] == 1


# --- 9. Telemetry failure does not break the request --------------------------------------------


def test_broken_event_logger_never_breaks_the_websocket(app_client, monkeypatch):
    def _broken_sink(event):
        raise RuntimeError("disk full")

    monkeypatch.setattr(app_client.app.state.event_logger, "sink", _broken_sink)
    session_id = create_session(app_client)["session_id"]
    with app_client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, BASE_UTTERANCE, 0)
        decision = ws.receive_json()  # must still arrive over the live socket
        assert decision["payload"]["decision"] == "WAIT"
        send_chunk(ws, 1, BASE_UTTERANCE + ", for 30 guests", 500)
        receive_retrieve_turn(ws)  # the whole turn still completes normally


# --- 10. Session/corpus isolation in telemetry --------------------------------------------------


def test_events_endpoint_never_leaks_another_sessions_events(app_client):
    alpha_session = create_session(app_client, "alpha")["session_id"]
    beta_session = create_session(app_client, "beta")["session_id"]

    with app_client.websocket_connect(f"/session/{alpha_session}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "hello", 0)
        ws.receive_json()

    with app_client.websocket_connect(f"/session/{beta_session}/stream?token={API_KEY}") as ws:
        send_chunk(ws, 0, "hello", 0)
        ws.receive_json()

    alpha_events = get_events_json(app_client, alpha_session)
    beta_events = get_events_json(app_client, beta_session)
    assert alpha_events and beta_events
    assert all(e["session_id"] == alpha_session for e in alpha_events)
    assert all(e["session_id"] == beta_session for e in beta_events)
