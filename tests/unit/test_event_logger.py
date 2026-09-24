"""Async Telemetry Event Logger (REQ-OBS-02/04; docs/TELEMETRY.md §4)."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.events import EventType, TelemetryEvent
from app.telemetry.event_logger import EventLogger
from tests.fakes import FIXTURE_CORPORA, make_settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(FIXTURE_CORPORA, tmp_path, SQLITE_PATH=":memory:")


@pytest.fixture
async def logger(settings):
    log = EventLogger(settings, flush_interval_s=0.02)
    await log.start()
    yield log
    await log.stop()


def _event(
    session_id: str, event_type: EventType = EventType.RETRIEVAL_DECISION, **payload
) -> TelemetryEvent:
    return TelemetryEvent(
        session_id=session_id, trace_id="trc-1", event_type=event_type, payload=payload
    )


# --- 4. async queue is non-blocking ---------------------------------------------------------------


def test_sink_is_synchronous_and_fast(logger):
    """REQ-OBS-04: the enqueue call itself must be a cheap in-memory operation - no disk I/O, no
    awaiting, safe to call from any synchronous context."""
    started = time.perf_counter()
    for _ in range(200):
        logger.sink(_event("sess_1"))
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 50  # 200 calls in well under 50ms - a single call is far below the 5ms cap


async def test_sink_never_raises_even_with_a_full_or_broken_queue(logger, monkeypatch):
    """9. telemetry failure does not break the request: sink() swallows any internal failure."""

    def _broken_put(*args, **kwargs):
        raise RuntimeError("queue is broken")

    monkeypatch.setattr(logger._queue, "put_nowait", _broken_put)
    logger.sink(_event("sess_1"))  # must not raise


# --- persistence / batching -------------------------------------------------------------------


async def test_events_are_persisted_and_queryable_after_flush(logger):
    logger.sink(_event("sess_1", EventType.RETRIEVAL_DECISION, decision="RETRIEVE"))
    await logger.flush()
    events = await logger.query("sess_1")
    assert len(events) == 1
    assert events[0].event_type == EventType.RETRIEVAL_DECISION
    assert events[0].payload["decision"] == "RETRIEVE"


async def test_events_persist_without_an_explicit_flush_via_the_background_timer(logger):
    logger.sink(_event("sess_1"))
    await asyncio.sleep(0.1)  # well past the 0.02s flush interval
    events = await logger.query("sess_1")
    assert len(events) == 1


async def test_query_with_no_events_returns_empty_list(logger):
    assert await logger.query("nonexistent_session") == []


# --- 5. telemetry ordering where specified ------------------------------------------------------


async def test_events_are_returned_in_emission_order(logger):
    for i in range(10):
        logger.sink(_event("sess_1", seq=i))
    await logger.flush()
    events = await logger.query("sess_1")
    assert [e.payload["seq"] for e in events] == list(range(10))


# --- 2. event payload/schema correctness ---------------------------------------------------------


async def test_persisted_event_round_trips_full_envelope(logger):
    original = TelemetryEvent(
        session_id="sess_1",
        trace_id="trc-42",
        event_type=EventType.CITATION_CREATED,
        payload={"chunk_id": "c1", "doc_id": "orion_hall", "section": "§3", "claim_text": "x"},
    )
    logger.sink(original)
    await logger.flush()
    (stored,) = await logger.query("sess_1")
    assert stored.event_id == original.event_id
    assert stored.session_id == original.session_id
    assert stored.trace_id == original.trace_id
    assert stored.event_type == original.event_type
    assert stored.payload == original.payload
    assert stored.timestamp == original.timestamp


# --- 10. session/corpus isolation in telemetry ----------------------------------------------------


async def test_query_scoped_to_session_id_never_leaks_another_sessions_events(logger):
    logger.sink(_event("sess_a", note="a"))
    logger.sink(_event("sess_b", note="b"))
    await logger.flush()

    events_a = await logger.query("sess_a")
    events_b = await logger.query("sess_b")

    assert {e.payload["note"] for e in events_a} == {"a"}
    assert {e.payload["note"] for e in events_b} == {"b"}


async def test_query_filters_by_trace_id_and_event_type(logger):
    logger.sink(
        TelemetryEvent(
            session_id="sess_1", trace_id="t1", event_type=EventType.RETRIEVAL_DECISION, payload={}
        )
    )
    logger.sink(
        TelemetryEvent(
            session_id="sess_1", trace_id="t2", event_type=EventType.RERANK_COMPLETED, payload={}
        )
    )
    await logger.flush()

    by_trace = await logger.query("sess_1", trace_id="t1")
    assert len(by_trace) == 1 and by_trace[0].trace_id == "t1"

    by_type = await logger.query("sess_1", event_type="RERANK_COMPLETED")
    assert len(by_type) == 1 and by_type[0].event_type == EventType.RERANK_COMPLETED


# --- 9. telemetry failure does not break the request (flush-level) -------------------------------


async def test_flush_failure_is_swallowed_and_does_not_raise(logger, monkeypatch):
    import app.telemetry.event_logger as event_logger_module

    def _broken_insert(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(event_logger_module, "_insert_batch", _broken_insert)
    logger.sink(_event("sess_1"))
    await logger.flush()  # must not raise despite the write failing internally


async def test_stop_is_safe_to_call_before_start_completes_any_work(tmp_path):
    """A logger that is started and immediately stopped (no events, connection barely
    established) must shut down cleanly - the same lifecycle every test's app_client fixture
    exercises via the real app lifespan."""
    log = EventLogger(make_settings(FIXTURE_CORPORA, tmp_path, SQLITE_PATH=":memory:"))
    await log.start()
    await log.stop()
