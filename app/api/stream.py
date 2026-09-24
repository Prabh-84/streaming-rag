"""WS /session/{id}/stream (docs/API.md §3): the live transcript path.

Inbound TRANSCRIPT_CHUNK frames are order-resolved into the session buffer (REQ-STREAM-01), each
newly-committed chunk runs one Retrieval Controller pass (app.controller.retrieval_controller),
and every event that pass produces is delivered back over this same socket using the standard
TelemetryEvent envelope (docs/API.md §4) — no separate event schema.

Events are queued (asyncio.Queue) rather than sent directly from the controller/retriever call
stack: EventSink.__call__ is synchronous (app.core.events), so a background sender task drains the
queue and performs the actual async `websocket.send_json`. This is the same async-queue shape
REQ-OBS-04 specifies for the eventual Phase 8 SQLite writer, applied here to the live socket.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.controller.retrieval_controller import on_chunk
from app.core.config import get_settings
from app.core.events import EventType, TelemetryEvent
from app.generation.streaming_generator import stream_answer
from app.models import new_id
from app.models.transcript_chunk import TranscriptChunk
from app.retrieval.hybrid import HybridRetriever
from app.session.session_store import SessionExpiredError, SessionStore, UnknownSessionError

router = APIRouter()

WS_AUTH_FAILED = 4401
WS_UNKNOWN_SESSION = 4404


def _error_event(session_id: str, stage: str, error_type: str, message: str) -> TelemetryEvent:
    return TelemetryEvent(
        session_id=session_id,
        trace_id=new_id(),
        event_type=EventType.ERROR,
        payload={"stage": stage, "error_type": error_type, "message": message, "recoverable": True},
    )


def _parse_transcript_chunk(raw: Any) -> TranscriptChunk:
    if not isinstance(raw, dict) or raw.get("event_type") != "TRANSCRIPT_CHUNK":
        raise ValueError("expected a frame with event_type == 'TRANSCRIPT_CHUNK'")
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("frame is missing a 'payload' object")
    return TranscriptChunk.model_validate(payload)


async def _drain_events(websocket: WebSocket, queue: asyncio.Queue[TelemetryEvent]) -> None:
    while True:
        event = await queue.get()
        await websocket.send_json(event.model_dump(mode="json"))


@router.websocket("/session/{session_id}/stream")
async def stream_endpoint(websocket: WebSocket, session_id: str) -> None:
    # A custom close code (4401/4404, docs/API.md §3) only exists in a WS Close frame, which only
    # exists after a successful upgrade — closing before accept() can only surface as a generic
    # HTTP-level rejection to the client, not the documented code. So every rejection path here
    # accepts first, then immediately closes with the specific code.
    settings = get_settings()
    if websocket.query_params.get("token") != settings.api_key:  # REQ-SEC-06
        await websocket.accept()
        await websocket.close(code=WS_AUTH_FAILED)
        return

    store: SessionStore = websocket.app.state.session_store
    retriever: HybridRetriever = websocket.app.state.hybrid_retriever

    try:
        record = await store.get(session_id)
    except (UnknownSessionError, SessionExpiredError):
        await websocket.accept()
        await websocket.close(code=WS_UNKNOWN_SESSION)
        return

    await websocket.accept()
    queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()

    def sink(event: TelemetryEvent) -> None:
        record.event_log.append(event)
        queue.put_nowait(event)

    sender_task = asyncio.create_task(_drain_events(websocket, queue))
    try:
        if await store.mark_connected(session_id):  # REQ-STREAM-03
            sink(
                TelemetryEvent(
                    session_id=session_id,
                    trace_id=new_id(),
                    event_type=EventType.SESSION_RESYNC,
                    payload={
                        "latest_answer_version": len(record.answer_versions) or None,
                        "entities": dict(record.entities),
                        "last_seq": record.buffer[-1].seq if record.buffer else None,
                    },
                )
            )

        while True:
            raw = await websocket.receive_json()
            try:
                chunk = _parse_transcript_chunk(raw)
            except (ValidationError, ValueError) as exc:
                sink(_error_event(session_id, "stream", "InvalidFrame", str(exc)))
                continue

            async with (
                record.lock
            ):  # serializes this session's turns; sessions never block each other
                trace_id = new_id()
                committed, sequence_gap = record.ingest_chunk(chunk, settings.reorder_window_ms)
                if sequence_gap:
                    sink(_error_event(session_id, "stream", "SequenceGap", "sequence_gap"))
                for committed_chunk in committed:
                    try:
                        decision = await on_chunk(
                            record,
                            committed_chunk,
                            trace_id=trace_id,
                            retriever=retriever,
                            sink=sink,
                        )
                        if decision.decision == "RETRIEVE":
                            # Phase 7: grounded generation runs after the controller's own
                            # decision/retrieval/fusion pipeline, over whatever evidence it
                            # produced - never a second retrieval trigger.
                            await stream_answer(
                                record,
                                decision.sub_queries,
                                decision.evidence,
                                decision.retrievals,
                                trace_id=trace_id,
                                sink=sink,
                            )
                    except Exception as exc:  # a stage failure degrades, never closes the socket
                        sink(_error_event(session_id, "controller", type(exc).__name__, str(exc)))
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task
