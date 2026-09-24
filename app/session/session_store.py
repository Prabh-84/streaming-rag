"""In-memory session store: lifecycle CRUD, TTL expiry, per-session serialization, and the
rolling controller/streaming state pseudocode 12.A calls `session.*`.

Scope note: session isolation (REQ-SESS-03), corpus_id immutability (REQ-CORPUS-02), and the
Retrieval Controller's rolling state (embedding_history, entities, wait_count,
has_retrieved_for_topic, topic_embedding, last_evidence, answer_versions) needed to run the
WebSocket stream, Phase 6 session refinement, and Phase 7 grounded generation. A full
claim-lifecycle ledger (REQ-SESS-02's contradiction/supersession/carry-forward bookkeeping) and
the SQLite crash-recovery mirror are not implemented here — see app.generation.streaming_generator
for the current, more limited claim tracking (AnswerVersion.new_claim_ids only).
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from enum import StrEnum

from app.core.config import Settings, get_settings
from app.core.events import EventSink, EventType, TelemetryEvent
from app.core.slots import validate_corpus_id
from app.models import new_id
from app.models.answer_version import AnswerVersion
from app.models.evidence import Evidence
from app.models.transcript_chunk import TranscriptChunk


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class UnknownSessionError(KeyError):
    """No session exists (or ever existed) with this session_id."""


class SessionExpiredError(KeyError):
    """The session existed but its TTL has elapsed (REQ-SEC-02) or it was explicitly closed."""


class SessionRecord:
    """One session's full runtime state. Never constructed directly outside SessionStore — use
    SessionStore.create() so every session is registered and lockable by session_id (REQ-SESS-03).
    """

    def __init__(self, session_id: str, corpus_id: str, created_at: datetime) -> None:
        self.session_id = session_id
        self.corpus_id = corpus_id  # immutable for the session's lifetime (PRD_TRD.md §7.8)
        self.created_at = created_at
        self.last_active_at = created_at
        self.status = SessionStatus.ACTIVE

        # Controller rolling state (pseudocode 12.A: session.embedding_history / .entities / ...)
        self.entities: dict[str, str] = {}
        self.embedding_history: list[tuple[float, ...]] = []
        self.wait_count: int = 0
        self.has_retrieved_for_topic: bool = False

        # Phase 6 session refinement (pseudocode 12.G): the embedding of whichever chunk most
        # recently triggered a RETRIEVE (full pipeline or refinement), and that turn's final
        # evidence set — reused as-is when a later turn introduces no new actionable information.
        self.topic_embedding: tuple[float, ...] | None = None
        self.last_evidence: list[Evidence] = []

        # Phase 7 grounded generation (pseudocode 12.I): committed answer versions, in order.
        # check_suppression's first-turn guard (REQ-SUPPRESS-02) also reads this list's
        # truthiness — unaffected by what element type it holds.
        self.answer_versions: list[AnswerVersion] = []

        # Ordered, gap-resolved transcript buffer (REQ-STREAM-01).
        self.buffer: list[TranscriptChunk] = []
        self._pending: dict[int, TranscriptChunk] = {}
        self._next_seq: int | None = None
        self._gap_opened_at_ms: int | None = None

        # WS lifecycle (REQ-STREAM-03: SESSION_RESYNC fires on reconnect, not on first connect).
        self.ever_connected: bool = False

        # In-memory event log (REQ-OBS-02/03 groundwork) — the Phase 8 async-queue/SQLite writer
        # replaces this; kept here now so GET /session/{id}/events has something to build on and
        # tests can inspect a session's telemetry without a live WebSocket client.
        self.event_log: list[TelemetryEvent] = []

        # Serializes processing within one session; sessions never block each other (original TRD
        # §9.6). A new lock per session, never a global one.
        self.lock = asyncio.Lock()

    def buffer_text(self) -> str:
        """Raw concatenation of committed chunks in order (REQ-STREAM-01: "buffer state matches
        concatenation at each step") — chunks are expected to carry their own spacing/punctuation,
        as real streaming transcript sources do."""
        return "".join(c.text_delta for c in self.buffer)

    def ingest_chunk(
        self, chunk: TranscriptChunk, reorder_window_ms: int
    ) -> tuple[list[TranscriptChunk], bool]:
        """Commit an incoming chunk into buffer order. Returns (newly_committed, sequence_gap).

        Out-of-order arrivals are held in a pending set and reordering-resolved for up to
        `reorder_window_ms` (measured against the chunks' own t_offset_ms, not wall-clock time, so
        replay stays deterministic); beyond that the pending set is committed as-is and
        `sequence_gap` is True (REQ-STREAM-01 failure behavior — logged ERROR: sequence_gap by the
        caller, since this module only manages state, not telemetry).
        """
        if self._next_seq is None:
            self._next_seq = chunk.seq

        if chunk.seq < self._next_seq:
            return [], False  # stale/duplicate: already committed, ignore

        if chunk.seq == self._next_seq:
            committed = [chunk]
            self.buffer.append(chunk)
            self._next_seq += 1
            while self._next_seq in self._pending:
                nxt = self._pending.pop(self._next_seq)
                self.buffer.append(nxt)
                committed.append(nxt)
                self._next_seq += 1
            if not self._pending:
                self._gap_opened_at_ms = None
            return committed, False

        # chunk.seq > self._next_seq: a gap exists.
        self._pending[chunk.seq] = chunk
        if self._gap_opened_at_ms is None:
            self._gap_opened_at_ms = chunk.t_offset_ms
        elapsed = chunk.t_offset_ms - self._gap_opened_at_ms
        if elapsed < reorder_window_ms:
            return [], False

        forced = sorted(self._pending.values(), key=lambda c: c.seq)
        self._pending.clear()
        self.buffer.extend(forced)
        self._next_seq = forced[-1].seq + 1
        self._gap_opened_at_ms = None
        return forced, True


class SessionStore:
    """Process-wide session registry. Every lookup is keyed exclusively by session_id
    (REQ-SESS-03) — there is no query path that can return another session's state."""

    def __init__(self, settings: Settings | None = None, *, sink: EventSink | None = None) -> None:
        self._settings = settings or get_settings()
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()
        # Phase 8 (REQ-OBS-01/02): SESSION_UPDATED for status transitions this store itself
        # detects (TTL expiry, explicit close) — decoupled from any one WebSocket connection, so
        # it fires even when the transition is noticed during an unrelated request.
        self._sink = sink

    async def create(self, corpus_id: str) -> SessionRecord:
        corpus_id = validate_corpus_id(corpus_id)
        record = SessionRecord(new_id(), corpus_id, datetime.now(UTC))
        async with self._lock:
            self._sessions[record.session_id] = record
        return record

    async def get(self, session_id: str) -> SessionRecord:
        """Raises UnknownSessionError / SessionExpiredError rather than ever returning another
        session's data — callers must not fall back to a different key on failure."""
        async with self._lock:
            record = self._sessions.get(session_id)
        if record is None:
            raise UnknownSessionError(session_id)
        if record.status == SessionStatus.CLOSED:
            raise SessionExpiredError(session_id)
        age = (datetime.now(UTC) - record.last_active_at).total_seconds()
        if age > self._settings.session_ttl_seconds:
            self._close_record(record, session_id)
            raise SessionExpiredError(session_id)
        return record

    async def touch(self, session_id: str) -> None:
        record = await self.get(session_id)
        record.last_active_at = datetime.now(UTC)

    async def close(self, session_id: str) -> None:
        async with self._lock:
            record = self._sessions.get(session_id)
        if record is not None and record.status != SessionStatus.CLOSED:
            self._close_record(record, session_id)

    def _close_record(self, record: SessionRecord, session_id: str) -> None:
        record.status = SessionStatus.CLOSED
        event = TelemetryEvent(
            session_id=session_id,
            trace_id=new_id(),
            event_type=EventType.SESSION_UPDATED,
            payload={"field": "status", "old_value": "active", "new_value": "closed"},
        )
        record.event_log.append(event)
        if self._sink is not None:
            # telemetry must never break a session-lifecycle transition
            with contextlib.suppress(Exception):
                self._sink(event)

    async def mark_connected(self, session_id: str) -> bool:
        """Returns True iff a WebSocket has already been attached to this session before (i.e.
        this call is a reconnect) — the trigger for emitting SESSION_RESYNC (REQ-STREAM-03)."""
        record = await self.get(session_id)
        was_connected = record.ever_connected
        record.ever_connected = True
        return was_connected

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        """Side-effect-free existence check (no expiry check, no status mutation) - unlike
        `get()`, which is deliberately unsuitable here since it can itself mutate `status` on a
        TTL-expiry detection. Used by GET /session/{id}/events (REQ-OBS-03) to distinguish a
        session_id this process never created at all from one that simply has no matching events."""
        return session_id in self._sessions
