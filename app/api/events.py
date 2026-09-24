"""GET /session/{id}/events (docs/API.md §6; REQ-OBS-03).

Full ordered event log for replay/audit, read from the same persistent store every pipeline stage
writes to via app.telemetry.event_logger.EventLogger — never from any one WebSocket connection's
own live-delivery queue, so this works whether or not a client is currently connected.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import require_api_key
from app.session.session_store import SessionStore
from app.telemetry.event_logger import EventLogger

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/session/{session_id}/events", response_model=None)
async def get_session_events(
    session_id: str,
    request: Request,
    trace_id: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    since: str | None = Query(default=None),
    fmt: str | None = Query(default=None, alias="format"),
) -> StreamingResponse | JSONResponse:
    event_logger: EventLogger = request.app.state.event_logger
    await event_logger.flush()  # so an event enqueued moments ago is already queryable
    events = await event_logger.query(
        session_id, trace_id=trace_id, event_type=event_type, since=since
    )

    if not events:
        # A session this process never created and has no recorded history for is unknown; one
        # that has history but no longer exists in the live store has simply outlived it - its
        # log is still exactly what REQ-OBS-03 asks to replay, so that alone is never a 404.
        store: SessionStore = request.app.state.session_store
        if session_id not in store:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session")

    if fmt == "json":
        return JSONResponse(content=[json.loads(e.model_dump_json()) for e in events])

    async def _sse() -> AsyncIterator[str]:
        for event in events:
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")
