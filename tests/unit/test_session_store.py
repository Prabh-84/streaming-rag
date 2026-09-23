"""SessionStore: lifecycle, TTL, isolation, and chunk ordering/reordering (REQ-SESS-03,
REQ-STREAM-01, REQ-SEC-02)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.models.transcript_chunk import TranscriptChunk
from app.session.session_store import (
    SessionExpiredError,
    SessionStatus,
    SessionStore,
    UnknownSessionError,
)
from tests.fakes import make_settings


def chunk(seq: int, text: str, t_offset_ms: int, is_final: bool = False) -> TranscriptChunk:
    return TranscriptChunk(seq=seq, text_delta=text, t_offset_ms=t_offset_ms, is_final=is_final)


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(make_settings(tmp_path, tmp_path, SESSION_TTL_SECONDS=1800))


async def test_create_assigns_a_session_id_and_fixes_corpus_id(store):
    record = await store.create("alpha")
    assert record.session_id
    assert record.corpus_id == "alpha"
    assert record.status == SessionStatus.ACTIVE
    assert record.answer_versions == []
    assert record.entities == {}


async def test_get_returns_the_same_record(store):
    created = await store.create("alpha")
    fetched = await store.get(created.session_id)
    assert fetched is created


async def test_get_unknown_session_raises(store):
    with pytest.raises(UnknownSessionError):
        await store.get("sess_does_not_exist")


async def test_get_closed_session_raises_expired(store):
    record = await store.create("alpha")
    await store.close(record.session_id)
    with pytest.raises(SessionExpiredError):
        await store.get(record.session_id)


async def test_get_ttl_expired_session_raises_and_closes_it(store):
    record = await store.create("alpha")
    record.last_active_at = datetime.now(UTC) - timedelta(seconds=99999)
    with pytest.raises(SessionExpiredError):
        await store.get(record.session_id)
    assert record.status == SessionStatus.CLOSED


async def test_touch_refreshes_last_active_at(store):
    record = await store.create("alpha")
    stale = datetime.now(UTC) - timedelta(seconds=10)
    record.last_active_at = stale
    await store.touch(record.session_id)
    assert record.last_active_at > stale


# --- Session isolation (REQ-SESS-03) -----------------------------------------------------------


async def test_session_isolation(store):
    """A transcript/entity change in session A must never be visible from session B, even when
    both exist concurrently and share a corpus_id."""
    a = await store.create("alpha")
    b = await store.create("alpha")
    assert a.session_id != b.session_id

    a.ingest_chunk(chunk(0, "for Orion Hall", 0), 2000)
    a.entities["venue"] = "Orion Hall"
    a.wait_count = 4
    a.has_retrieved_for_topic = True

    fetched_b = await store.get(b.session_id)
    assert fetched_b is b
    assert fetched_b.entities == {}
    assert fetched_b.wait_count == 0
    assert fetched_b.has_retrieved_for_topic is False
    assert fetched_b.buffer == []


async def test_corpus_isolation_across_sessions(store):
    """Two sessions on different corpora never share or leak corpus_id."""
    a = await store.create("alpha")
    b = await store.create("beta")
    assert a.corpus_id == "alpha"
    assert b.corpus_id == "beta"
    assert (await store.get(a.session_id)).corpus_id == "alpha"
    assert (await store.get(b.session_id)).corpus_id == "beta"


async def test_sessions_have_independent_locks(store):
    a = await store.create("alpha")
    b = await store.create("beta")
    assert a.lock is not b.lock


# --- Transcript buffering and ordering (REQ-STREAM-01) -----------------------------------------


async def test_incremental_chunks_are_appended_in_order(store):
    record = await store.create("alpha")
    for i, text in enumerate(["Hello ", "world, ", "how are ", "you?"]):
        committed, gap = record.ingest_chunk(chunk(i, text, i * 500), 2000)
        assert gap is False
        assert committed == [record.buffer[-1]]
    assert record.buffer_text() == "Hello world, how are you?"


async def test_buffer_state_matches_concatenation_at_each_step(store):
    """BENCH-03: buffer state matches concatenation at each step."""
    record = await store.create("alpha")
    pieces = ["I ", "need ", "a venue ", "for 30 ", "guests."]
    running = ""
    for i, text in enumerate(pieces):
        record.ingest_chunk(chunk(i, text, i * 400), 2000)
        running += text
        assert record.buffer_text() == running


async def test_out_of_order_chunk_is_held_and_committed_once_the_gap_fills(store):
    record = await store.create("alpha")
    committed0, gap0 = record.ingest_chunk(chunk(0, "A", 0), 2000)
    committed2, gap2 = record.ingest_chunk(chunk(2, "C", 400), 2000)  # arrives early, out of order
    assert committed0 == [record.buffer[0]] and gap0 is False
    assert committed2 == [] and gap2 is False  # held back, not yet committed
    assert record.buffer_text() == "A"

    committed1, gap1 = record.ingest_chunk(chunk(1, "B", 800), 2000)
    assert gap1 is False
    assert [c.text_delta for c in committed1] == ["B", "C"]  # the gap-filler plus the held chunk
    assert record.buffer_text() == "ABC"


async def test_gap_beyond_reorder_window_is_applied_as_is_and_flagged(store):
    """REQ-STREAM-01 failure behavior: beyond the reorder window, apply as-is and signal the gap
    (logged as ERROR: sequence_gap by the caller)."""
    record = await store.create("alpha")
    record.ingest_chunk(chunk(0, "A", 0), 2000)
    record.ingest_chunk(chunk(2, "C", 100), 2000)  # seq 1 never arrives; gap opens at t=100

    committed, gap = record.ingest_chunk(chunk(3, "D", 100 + 1999), 2000)  # just inside the window
    assert gap is False
    assert committed == []

    committed, gap = record.ingest_chunk(chunk(4, "E", 100 + 2000), 2000)  # window elapsed -> flush
    assert gap is True
    assert [c.text_delta for c in committed] == ["C", "D", "E"]  # flushed as-is, seq 1 skipped
    assert record.buffer_text() == "ACDE"


async def test_gap_timeout_uses_chunk_timestamps_not_wall_clock(store):
    """Deterministic replay: two chunks passed back-to-back in real time, but far apart by their
    own t_offset_ms, must behave identically to a real 2s wait."""
    record = await store.create("alpha")
    record.ingest_chunk(chunk(0, "A", 0), 2000)
    record.ingest_chunk(chunk(2, "C", 0), 2000)  # gap opens at virtual t=0
    committed, gap = record.ingest_chunk(chunk(3, "D", 5000), 2000)  # "5s later" with no real delay
    assert gap is True
    assert [c.text_delta for c in committed] == ["C", "D"]


async def test_duplicate_or_stale_chunk_is_ignored(store):
    record = await store.create("alpha")
    record.ingest_chunk(chunk(0, "A", 0), 2000)
    record.ingest_chunk(chunk(1, "B", 100), 2000)
    committed, gap = record.ingest_chunk(chunk(0, "A-again", 200), 2000)  # replay of seq 0
    assert committed == [] and gap is False
    assert record.buffer_text() == "AB"  # never re-applied
