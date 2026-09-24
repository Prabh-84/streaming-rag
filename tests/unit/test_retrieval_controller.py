"""Retrieval Controller: WAIT/RETRIEVE/NO_RETRIEVAL decisions (REQ-CTRL-01/02/03, REQ-STREAM-*).

Uses a hash-based FakeEmbedder (deterministic, no model download) and a fake HybridRetriever so
these are pure unit tests of the controller's decision logic — never dependent on a real Qdrant
server (per the Phase 3 brief).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.controller import retrieval_controller as ctrl
from app.controller.entity_extraction import get_corpus_matcher
from app.core.config import get_settings
from app.core.events import RetrievalTrigger
from app.core.slots import get_slot_schema
from app.models.retrieval_event import RetrievalResult
from app.models.transcript_chunk import TranscriptChunk
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse_bm25 import SparseIndexRegistry
from app.session.session_store import SessionStore
from scripts.ingest_corpus import ingest_corpus
from tests.fakes import FIXTURE_CORPORA, FakeDecomposer, FakeEmbedder, RecordingSink, make_settings

COMPOUND_TEXT = "What is the cancellation policy and the catering options for Orion Hall"


class FakeRetriever:
    """Records every call; never touches Qdrant/BM25 — controller unit tests must not depend on
    the real retrieval backends."""

    def __init__(self) -> None:
        self.calls: list = []

    async def retrieve(self, request, *, sink=None) -> RetrievalResult:
        self.calls.append(request)
        return RetrievalResult(sub_query_id=request.sub_query_id, corpus_id=request.corpus_id)


@pytest.fixture(autouse=True)
def _use_fixture_corpus_root(monkeypatch):
    monkeypatch.setattr(get_settings(), "corpus_root", str(FIXTURE_CORPORA))
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()
    yield
    get_slot_schema.cache_clear()
    get_corpus_matcher.cache_clear()


@pytest.fixture
def settings(tmp_path):
    return make_settings(FIXTURE_CORPORA, tmp_path, STABILITY_THRESHOLD=0.90, MAX_WAIT_CHUNKS=6)


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def session(settings):
    store = SessionStore(settings)
    return await store.create("alpha")


def chunk(seq: int, text: str, t_offset_ms: int, is_final: bool = False) -> TranscriptChunk:
    return TranscriptChunk(seq=seq, text_delta=text, t_offset_ms=t_offset_ms, is_final=is_final)


async def run_chunk(
    session,
    text,
    t_offset_ms,
    *,
    retriever,
    embedder,
    settings,
    llm=None,
    is_final=False,
    sink=None,
    seq=None,
):
    c = chunk(len(session.buffer) if seq is None else seq, text, t_offset_ms, is_final)
    committed, _ = session.ingest_chunk(c, settings.reorder_window_ms)
    assert committed == [c]
    return await ctrl.on_chunk(
        session,
        c,
        trace_id="trc_1",
        retriever=retriever,
        embedder=embedder,
        llm=llm or FakeDecomposer(sub_queries=None),
        settings=settings,
        sink=sink,
    )


# --- compute_stability: pure math, no mocking -------------------------------------------------


def test_compute_stability_identical_vectors_is_one():
    v = (1.0, 0.0, 0.0)
    assert ctrl.compute_stability([v, v], 2) == pytest.approx(1.0)


def test_compute_stability_orthogonal_vectors_is_zero():
    assert ctrl.compute_stability([(1.0, 0.0), (0.0, 1.0)], 2) == pytest.approx(0.0)


def test_compute_stability_needs_at_least_two_embeddings():
    assert ctrl.compute_stability([], 2) == 0.0
    assert ctrl.compute_stability([(1.0, 0.0)], 2) == 0.0


def test_compute_stability_window_limits_history_considered():
    # window=2 only looks at the last two entries, regardless of how much history exists.
    unstable_then_stable = [(1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]
    assert ctrl.compute_stability(unstable_then_stable, 2) == pytest.approx(1.0)


async def test_retrieve_end_to_end_without_mocking_stability(
    session, retriever, embedder, settings
):
    """The real compute_stability + FakeEmbedder pipeline, no monkeypatching: a long, mostly-shared
    transcript where the second chunk merely appends the actionable entity naturally clears
    STABILITY_THRESHOLD (high word overlap -> high hash-embedding cosine similarity) while also
    supplying the first actionable entity delta."""
    base = (
        "I am planning a corporate event next month and I need to book a venue "
        "that can comfortably fit everyone"
    )
    await run_chunk(session, base, 0, retriever=retriever, embedder=embedder, settings=settings)
    decision = await run_chunk(
        session,
        base + ", for 30 guests",
        400,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert decision.decision == ctrl.RETRIEVE
    assert len(retriever.calls) == 1


# --- WAIT ----------------------------------------------------------------------------------


async def test_wait_on_topic_change_between_chunks(session, retriever, embedder, settings):
    """5. actionable delta without stability: two chunks about unrelated topics with different
    entities should not have high inter-chunk stability, and even if they did, the FIRST chunk
    alone can never be stable (needs >=2 embeddings) -> WAIT."""
    decision = await run_chunk(
        session, "hello there", 0, retriever=retriever, embedder=embedder, settings=settings
    )
    assert decision.decision == ctrl.WAIT
    assert decision.reason == "intent_unstable"
    assert retriever.calls == []


async def test_wait_below_stability_threshold_even_with_actionable_entity(
    session, retriever, embedder, settings, monkeypatch
):
    """4. intent stability below threshold -> WAIT, even though an entity is present."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.10)
    decision = await run_chunk(
        session, "Orion Hall", 0, retriever=retriever, embedder=embedder, settings=settings
    )
    assert decision.decision == ctrl.WAIT
    assert decision.reason == "intent_unstable"
    assert retriever.calls == []


async def test_wait_when_stable_but_no_actionable_entity(
    session, retriever, embedder, settings, monkeypatch
):
    """6. stability at/above threshold without an actionable delta -> WAIT (a stable-but-empty
    clause, e.g. "I need to plan a...", per PRD_TRD.md §5.4)."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    decision = await run_chunk(
        session, "I need to plan a", 0, retriever=retriever, embedder=embedder, settings=settings
    )
    assert decision.decision == ctrl.WAIT
    assert decision.reason == "intent_unstable"  # pseudocode 12.A always logs this label here
    assert retriever.calls == []


# --- RETRIEVE --------------------------------------------------------------------------------


async def test_retrieve_on_stability_plus_actionable_delta(
    session, retriever, embedder, settings, monkeypatch
):
    """7. stability + actionable delta -> RETRIEVE."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    decision = await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert decision.decision == ctrl.RETRIEVE
    assert decision.trigger == RetrievalTrigger.PROVISIONAL
    assert len(retriever.calls) == 1
    assert retriever.calls[0].text == session.buffer_text()
    assert retriever.calls[0].corpus_id == "alpha"
    assert session.has_retrieved_for_topic is True


async def test_final_chunk_uses_final_trigger(session, retriever, embedder, settings, monkeypatch):
    """8. final utterance retrieval behavior: is_final=True labels the trigger "final"."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    decision = await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
        is_final=True,
    )
    assert decision.decision == ctrl.RETRIEVE
    assert decision.trigger == RetrievalTrigger.FINAL
    assert retriever.calls[0].trigger == RetrievalTrigger.FINAL


async def test_early_retrieval_before_final_transcript_completion(
    session, retriever, embedder, settings, monkeypatch
):
    """9. Provisional RETRIEVE fires on a non-final chunk, strictly before the eventual final
    chunk's t_offset_ms — REQ-STREAM-02's "no fake streaming" timing assertion."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    decision = await run_chunk(
        session,
        "Orion Hall for 30 guests",
        800,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
        is_final=False,
    )
    final_utterance_offset_ms = 2100
    assert decision.decision == ctrl.RETRIEVE
    assert decision.trigger == RetrievalTrigger.PROVISIONAL
    assert retriever.calls[0].t_offset_ms == 800 < final_utterance_offset_ms


async def test_duplicate_non_actionable_chunks_do_not_cause_repeated_retrieval(
    session, retriever, embedder, settings, monkeypatch
):
    """12. Once retrieved for a topic, a later chunk with no new entity delta WAITs instead of
    re-firing RETRIEVE (REQ-CTRL-01's "no_new_entity" branch)."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    first = await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert first.decision == ctrl.RETRIEVE
    assert len(retriever.calls) == 1

    second = await run_chunk(
        session, " please", 400, retriever=retriever, embedder=embedder, settings=settings
    )
    assert second.decision == ctrl.WAIT
    assert second.reason == "no_new_entity"
    assert len(retriever.calls) == 1  # not called again


async def test_new_entity_after_retrieval_can_retrigger_retrieve(
    session, retriever, embedder, settings, monkeypatch
):
    """A genuinely new entity delta after an initial RETRIEVE is still allowed to retrieve again
    ("only new chunks that change entity_state re-trigger", original TRD §6.1)."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    second = await run_chunk(
        session,
        " and the cancellation policy for Lumen Pavilion",
        400,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert second.decision == ctrl.RETRIEVE
    assert len(retriever.calls) == 2


# --- forced-after-max-wait ---------------------------------------------------------------------


async def test_forced_retrieve_after_max_wait_chunks(session, retriever, embedder, settings):
    """REQ-CTRL-01 failure behavior: wait_count > MAX_WAIT_CHUNKS forces RETRIEVE using
    best-available partial intent — the (max_wait_chunks + 1)th consecutive WAIT-eligible chunk."""
    decisions = []
    for i in range(settings.max_wait_chunks + 1):
        decisions.append(
            await run_chunk(
                session,
                f"um so like {i} ",
                i * 100,
                retriever=retriever,
                embedder=embedder,
                settings=settings,
            )
        )
    assert [d.decision for d in decisions[:-1]] == [ctrl.WAIT] * settings.max_wait_chunks
    assert decisions[-1].decision == ctrl.RETRIEVE
    assert decisions[-1].trigger == RetrievalTrigger.FORCED_AFTER_MAX_WAIT
    assert len(retriever.calls) == 1
    assert session.wait_count == 0  # reset after the forced retrieve

    # A subsequent still-non-actionable chunk starts a fresh WAIT cycle, not another force-fire.
    next_decision = await run_chunk(
        session,
        "um again",
        settings.max_wait_chunks * 100,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert next_decision.decision == ctrl.WAIT
    assert len(retriever.calls) == 1


# --- NO_RETRIEVAL / suppression ------------------------------------------------------------


async def test_no_retrieval_for_presentation_only_request_with_prior_answer(
    session, retriever, embedder, settings
):
    """10. formatting-only request -> NO_RETRIEVAL, and the retriever is never called (13/15)."""
    session.answer_versions.append("av_1")  # simulates a prior answer existing
    decision = await run_chunk(
        session,
        "please repeat that in two bullets",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert decision.decision == ctrl.NO_RETRIEVAL
    assert decision.reason == "presentation_only"
    assert retriever.calls == []


async def test_first_turn_presentation_pattern_is_not_suppressed(
    session, retriever, embedder, settings
):
    """11. first-turn suppression guard: no prior answer -> falls through to normal WAIT logic,
    never NO_RETRIEVAL."""
    decision = await run_chunk(
        session, "please repeat that", 0, retriever=retriever, embedder=embedder, settings=settings
    )
    assert decision.decision != ctrl.NO_RETRIEVAL


# --- retrieval invocation (13/14/15) -------------------------------------------------------


async def test_retrieval_invoked_only_on_retrieve(
    session, retriever, embedder, settings, monkeypatch
):
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert len(retriever.calls) == 1


async def test_retrieval_not_invoked_on_wait(session, retriever, embedder, settings):
    await run_chunk(session, "um", 0, retriever=retriever, embedder=embedder, settings=settings)
    assert retriever.calls == []


async def test_retrieval_not_invoked_on_no_retrieval(session, retriever, embedder, settings):
    session.answer_versions.append("av_1")
    await run_chunk(
        session, "please repeat that", 0, retriever=retriever, embedder=embedder, settings=settings
    )
    assert retriever.calls == []


# --- telemetry -------------------------------------------------------------------------------


async def test_telemetry_events_and_trace_id_propagation(
    session, retriever, embedder, settings, monkeypatch
):
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    sink = RecordingSink()
    decision = await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
        sink=sink,
    )
    assert decision.decision == ctrl.RETRIEVE
    # A RETRIEVE turn now emits its decision, then one SUBQUERY_CREATED per sub-query (Phase 4) —
    # exactly one here, since this text carries no compound signal.
    decision_event, subquery_event = sink.events
    assert decision_event.session_id == session.session_id
    assert decision_event.trace_id == "trc_1"
    assert decision_event.event_type == "RETRIEVAL_DECISION"
    assert decision_event.payload == {
        "decision": "RETRIEVE",
        "trigger": "provisional",
        "reason": "provisional",
    }
    assert subquery_event.event_type == "SUBQUERY_CREATED"
    assert subquery_event.payload["text"] == "Orion Hall for 30 guests"
    assert subquery_event.payload["intent_label"] == "single"


async def test_wait_telemetry_has_no_trigger(session, retriever, embedder, settings):
    sink = RecordingSink()
    await run_chunk(
        session, "um", 0, retriever=retriever, embedder=embedder, settings=settings, sink=sink
    )
    (event,) = sink.events
    assert event.payload["decision"] == "WAIT"
    assert event.payload["trigger"] is None
    assert event.payload["reason"] == "intent_unstable"


# --- deterministic replay (23) -----------------------------------------------------------------


async def test_controller_deterministic_replay(settings):
    """Same chunk sequence, same timestamps, two independent sessions -> identical decisions."""
    script = [(0, "I need a venue "), (500, "in Orion Hall "), (1000, "for 30 guests")]

    async def replay():
        store = SessionStore(settings)
        session = await store.create("alpha")
        retriever = FakeRetriever()
        embedder = FakeEmbedder()
        decisions = []
        for seq, (t_offset_ms, text) in enumerate(script):
            c = chunk(seq, text, t_offset_ms)
            committed, _ = session.ingest_chunk(c, settings.reorder_window_ms)
            for committed_chunk in committed:
                decisions.append(
                    await ctrl.on_chunk(
                        session,
                        committed_chunk,
                        trace_id="t",
                        retriever=retriever,
                        embedder=embedder,
                        settings=settings,
                    )
                )
        return [(d.decision, d.reason, d.trigger) for d in decisions]

    first = await replay()
    second = await replay()
    assert first == second


# --- G. Retrieval integration (Phase 4: decomposition -> concurrent retrieval) ------------------


async def test_compound_request_calls_retriever_once_per_subquery_with_correct_triggers(
    session, retriever, embedder, settings, monkeypatch
):
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    llm = FakeDecomposer(
        sub_queries=[
            {
                "text": "What is the cancellation policy for Orion Hall?",
                "intent_label": "cancellation",
            },
            {"text": "What are the catering options for Orion Hall?", "intent_label": "catering"},
        ]
    )
    decision = await run_chunk(
        session,
        COMPOUND_TEXT,
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
        llm=llm,
    )
    assert decision.decision == ctrl.RETRIEVE
    assert len(retriever.calls) == 2
    assert len(decision.sub_query_ids) == 2
    assert len(decision.retrievals) == 2

    first, second = retriever.calls
    assert first.text == "What is the cancellation policy for Orion Hall?"
    assert first.trigger == RetrievalTrigger.PROVISIONAL  # keeps the controller-level trigger
    assert second.text == "What are the catering options for Orion Hall?"
    assert second.trigger == RetrievalTrigger.MULTI_INTENT  # REQ-OBS-05


async def test_compound_request_preserves_corpus_id_for_every_subquery(
    session, retriever, embedder, settings, monkeypatch
):
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    llm = FakeDecomposer(
        sub_queries=[
            {"text": "cancellation policy for Orion Hall", "intent_label": "a"},
            {"text": "catering options for Orion Hall", "intent_label": "b"},
        ]
    )
    await run_chunk(
        session,
        COMPOUND_TEXT,
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
        llm=llm,
    )
    assert len(retriever.calls) == 2
    assert all(request.corpus_id == "alpha" == session.corpus_id for request in retriever.calls)


async def test_compound_request_subqueries_are_retrieved_concurrently(
    session, embedder, settings, monkeypatch
):
    """Wall time for N sub-query retrievals must be ~one retrieval's latency, not the sum —
    the controller dispatches them via asyncio.gather, exactly as it already did for dense+sparse
    within a single sub-query (REQ-EVID-01)."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)

    class SlowRetriever:
        def __init__(self) -> None:
            self.calls = []

        async def retrieve(self, request, *, sink=None) -> RetrievalResult:
            self.calls.append(request)
            await asyncio.sleep(0.2)
            return RetrievalResult(sub_query_id=request.sub_query_id, corpus_id=request.corpus_id)

    retriever = SlowRetriever()
    llm = FakeDecomposer(
        sub_queries=[
            {"text": "cancellation policy for Orion Hall", "intent_label": "a"},
            {"text": "catering options for Orion Hall", "intent_label": "b"},
        ]
    )
    started = time.perf_counter()
    decision = await run_chunk(
        session,
        COMPOUND_TEXT,
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
        llm=llm,
    )
    elapsed = time.perf_counter() - started
    assert decision.decision == ctrl.RETRIEVE
    assert len(retriever.calls) == 2
    assert elapsed < 0.35  # ~max(0.2, 0.2), not the 0.4 sum


async def test_single_intent_request_still_calls_retriever_exactly_once(
    session, retriever, embedder, settings, monkeypatch
):
    """Regression guard: a normal, non-compound request must retrieve exactly once, exactly as
    it did before decomposition existed (Phase 3 behavior, now routed through the decomposer's
    trivial single-subquery path)."""
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)
    decision = await run_chunk(
        session,
        "Orion Hall for 30 guests",
        0,
        retriever=retriever,
        embedder=embedder,
        settings=settings,
    )
    assert decision.decision == ctrl.RETRIEVE
    assert len(retriever.calls) == 1
    assert retriever.calls[0].trigger == RetrievalTrigger.PROVISIONAL  # never multi_intent
    assert len(decision.sub_query_ids) == 1


@pytest.fixture
async def real_alpha_retriever(tmp_path):
    """The genuine Phase 2 HybridRetriever (real semaphore, real timeouts), over an in-memory
    Qdrant + BM25 index built from the actual alpha fixture corpus via a FakeEmbedder — proves
    Phase 4 dispatches into the *existing* retrieval primitive, not a reimplementation."""
    from qdrant_client import AsyncQdrantClient

    real_settings = make_settings(FIXTURE_CORPORA, tmp_path)
    embedder = FakeEmbedder()
    qdrant_client = AsyncQdrantClient(location=":memory:")
    await ingest_corpus(
        "alpha", settings=real_settings, embedder=embedder, qdrant_client=qdrant_client
    )
    dense = DenseIndex(qdrant_client, embedder, real_settings)
    sparse = SparseIndexRegistry(real_settings)
    yield HybridRetriever(dense, sparse, real_settings), real_settings, embedder
    await qdrant_client.close()


async def test_compound_request_end_to_end_with_real_hybrid_retriever(
    real_alpha_retriever, monkeypatch
):
    """Section G, full stack: real corpus, real HybridRetriever (bounded concurrency + per-mode
    timeouts unchanged from Phase 2/3), driven by a genuinely compound request."""
    real_retriever, real_settings, embedder = real_alpha_retriever
    monkeypatch.setattr(ctrl, "compute_stability", lambda history, window: 0.99)

    store = SessionStore(real_settings)
    session = await store.create("alpha")
    llm = FakeDecomposer(
        sub_queries=[
            {"text": "cancellation policy for Orion Hall", "intent_label": "cancellation"},
            {"text": "catering options for Orion Hall", "intent_label": "catering"},
        ]
    )

    decision = await run_chunk(
        session,
        COMPOUND_TEXT,
        0,
        retriever=real_retriever,
        embedder=embedder,
        settings=real_settings,
        llm=llm,
    )

    assert decision.decision == ctrl.RETRIEVE
    assert len(decision.retrievals) == 2
    for result in decision.retrievals:
        assert result.corpus_id == "alpha"  # no cross-corpus leakage
        assert result.dense or result.sparse  # the real corpus actually has relevant content
