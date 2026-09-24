"""Async Telemetry Event Logger (REQ-OBS-02/04; docs/TELEMETRY.md §4).

Every `TelemetryEvent`, from every pipeline stage in every session, is queued here — a fast,
synchronous, non-blocking `asyncio.Queue.put_nowait()` call, satisfying REQ-OBS-04's "never adding
more than 5ms to the critical path per event" — and periodically batched to SQLite by a single
background writer task. This is deliberately decoupled from any one WebSocket connection's own
live-delivery queue (`app.api.stream`'s per-connection `sink`): a session's full event history
must survive a client disconnect, and even be observable *while no client is connected at all*
(REQ-OBS-02's "100% of sessions have a complete, replayable trace"), e.g. a TTL expiry detected
during some other session's request (`SessionStore.get`) still needs to land somewhere.

Plain `sqlite3` via `asyncio.to_thread`, not an ORM or an async driver: REQ-OBS-04's non-blocking
guarantee is about the *enqueue* call on the request's own path, not the disk write itself, which
already runs off that path in the background writer loop — a single small table doesn't need more
machinery than that.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import threading
from pathlib import Path

import structlog

from app.core.config import Settings, get_settings
from app.core.events import EventType, TelemetryEvent

log = structlog.get_logger()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_session ON telemetry_events(session_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_trace ON telemetry_events(trace_id);
"""

_INSERT = (
    "INSERT OR IGNORE INTO telemetry_events "
    "(event_id, session_id, trace_id, event_type, timestamp, payload) VALUES (?, ?, ?, ?, ?, ?)"
)


def _connect(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _insert_batch(
    conn: sqlite3.Connection, lock: threading.Lock, events: list[TelemetryEvent]
) -> None:
    rows = [
        (
            e.event_id,
            e.session_id,
            e.trace_id,
            e.event_type.value,
            e.timestamp.isoformat(),
            json.dumps(e.payload),
        )
        for e in events
    ]
    with lock:
        conn.executemany(_INSERT, rows)
        conn.commit()


def _run_query(
    conn: sqlite3.Connection, lock: threading.Lock, sql: str, params: list[str]
) -> list[tuple]:
    with lock:
        return conn.execute(sql, params).fetchall()


def _close_conn(conn: sqlite3.Connection, lock: threading.Lock) -> None:
    """Closing must be serialized exactly like every other use of `conn` - closing it while an
    insert or query still holds the lock elsewhere is a real access-violation risk in sqlite3's C
    extension, not just a logical error."""
    with lock:
        conn.close()


def _row_to_event(row: tuple) -> TelemetryEvent:
    event_id, session_id, trace_id, event_type, timestamp, payload = row
    return TelemetryEvent(
        event_id=event_id,
        session_id=session_id,
        trace_id=trace_id,
        event_type=EventType(event_type),
        timestamp=timestamp,
        payload=json.loads(payload),
    )


class EventLogger:
    """One instance per process (`app.state.event_logger`). `sink` is what every pipeline stage's
    `EventSink` callback ultimately reaches; `start()`/`stop()` bracket the app lifespan."""

    _CONNECT_WAIT_TIMEOUT_S = 2.0  # safety bound for flush()/query() below, not a spec tunable

    def __init__(self, settings: Settings | None = None, *, flush_interval_s: float = 0.1) -> None:
        self._settings = settings or get_settings()
        self._flush_interval_s = flush_interval_s
        self._queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._conn: sqlite3.Connection | None = None
        # sqlite3's check_same_thread=False only makes the *object* usable from another thread —
        # it does not make concurrent access from multiple threads safe. The background flush
        # loop and query() (called from a request handler) each run on their own thread via
        # asyncio.to_thread, so every actual use of self._conn is serialized through this lock.
        self._db_lock = threading.Lock()
        # Set once _run()'s own connect step finishes (success or failure) - flush()/query() wait
        # on this rather than racing start()'s deliberately-not-awaited connect (see start()).
        self._connected = asyncio.Event()

    async def _wait_connected(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._connected.wait(), self._CONNECT_WAIT_TIMEOUT_S)

    def sink(self, event: TelemetryEvent) -> None:
        """REQ-OBS-04: enqueue only, never touches the disk. Never raises - telemetry must never
        break the pipeline stage that called it (the same convention every stage's own emit
        helper already follows for its own sink calls)."""
        try:
            self._queue.put_nowait(event)
        except Exception:
            log.exception("telemetry_enqueue_failed", event_type=event.event_type.value)

    async def start(self) -> None:
        """Returns immediately - the SQLite connection is established inside the background task
        itself, the same "never block the lifespan's own critical path" convention app.main's
        _warm_embedder/_warm_nlp already use. sink() never depends on self._conn being ready yet
        (it only ever touches the in-memory queue), so nothing is lost by connecting lazily here."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._flush_pending()
        if self._conn is not None:
            await asyncio.to_thread(_close_conn, self._conn, self._db_lock)
            self._conn = None

    async def _run(self) -> None:
        try:
            self._conn = await asyncio.to_thread(_connect, self._settings.sqlite_path)
        except Exception:
            log.exception("telemetry_connect_failed", db_path=self._settings.sqlite_path)
        finally:
            # Always set, even on failure: flush()/query() must not hang waiting for a connection
            # that will never come - they degrade to a no-op instead (self._conn stays None).
            self._connected.set()
        while True:
            await asyncio.sleep(self._flush_interval_s)
            await self._flush_pending()

    async def _flush_pending(self) -> None:
        batch: list[TelemetryEvent] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not batch or self._conn is None:
            return
        try:
            await asyncio.to_thread(_insert_batch, self._conn, self._db_lock, batch)
        except Exception:
            # A write failure degrades observability, never the request path that already
            # returned once its event was enqueued (REQ-OBS-04's spirit, same as every other
            # stage's "telemetry must never break X" convention).
            log.exception("telemetry_flush_failed", batch_size=len(batch))

    async def flush(self) -> None:
        """Test/on-demand hook: force an immediate flush without waiting for the timer. Waits
        (briefly, bounded) for _run()'s own connect step first, since start() deliberately does
        not await it."""
        await self._wait_connected()
        await self._flush_pending()

    async def query(
        self,
        session_id: str,
        *,
        trace_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
    ) -> list[TelemetryEvent]:
        """REQ-OBS-03: the full ordered event log for one session, optionally narrowed. Always
        scoped by session_id — the same "every lookup scoped by its key" discipline REQ-SESS-03
        applies to session state, applied here to telemetry (no query path can return another
        session's events). Runs off the request's own thread (asyncio.to_thread), both to honor
        REQ-OBS-04's non-blocking spirit and to keep every self._conn access serialized through
        the same lock the background writer uses."""
        await self._wait_connected()
        if self._conn is None:
            return []
        sql = (
            "SELECT event_id, session_id, trace_id, event_type, timestamp, payload "
            "FROM telemetry_events WHERE session_id = ?"
        )
        params: list[str] = [session_id]
        if trace_id is not None:
            sql += " AND trace_id = ?"
            params.append(trace_id)
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp ASC, rowid ASC"
        rows = await asyncio.to_thread(_run_query, self._conn, self._db_lock, sql, params)
        return [_row_to_event(row) for row in rows]
