"""TelemetryEvent envelope and event_type enum.

Schema is frozen in docs/TELEMETRY.md §1-2. This module defines the wire/storage shape only;
emission (async queue, batched SQLite writer) is implemented in app/telemetry/event_logger.py
(Phase 8) and is out of scope for this Phase 1 foundation module.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """All WebSocket server<->client frame types and internal TelemetryEvent types.

    See docs/TELEMETRY.md §2. SESSION_RESYNC is new (REQ-STREAM-03, resolution #4).
    """

    TRANSCRIPT_CHUNK = "TRANSCRIPT_CHUNK"  # inbound (client -> server)
    RETRIEVAL_DECISION = "RETRIEVAL_DECISION"
    SUBQUERY_CREATED = "SUBQUERY_CREATED"
    RETRIEVAL_STARTED = "RETRIEVAL_STARTED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    RERANK_COMPLETED = "RERANK_COMPLETED"
    CITATION_CREATED = "CITATION_CREATED"
    ANSWER_VERSION_CREATED = "ANSWER_VERSION_CREATED"
    ANSWER_DELTA = "ANSWER_DELTA"
    UNCERTAINTY = "UNCERTAINTY"
    ERROR = "ERROR"
    SESSION_UPDATED = "SESSION_UPDATED"
    SESSION_RESYNC = "SESSION_RESYNC"  # NEW: REQ-STREAM-03


class RetrievalTrigger(StrEnum):
    """RetrievalEvent.trigger enum (docs/TELEMETRY.md §2.1).

    MULTI_INTENT is new (REQ-OBS-05): tags a sub-query-level retrieval produced by decomposition,
    distinct from the controller-level trigger that fired the first sub-query's retrieval.
    """

    PROVISIONAL = "provisional"
    FINAL = "final"
    DELTA = "delta"
    FORCED_AFTER_MAX_WAIT = "forced_after_max_wait"
    MULTI_INTENT = "multi_intent"


class TelemetryEvent(BaseModel):
    """Standard event envelope (REQ-OBS-01). Shared by every telemetry record and every
    WebSocket server->client frame.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str

    model_config = {"frozen": True}
