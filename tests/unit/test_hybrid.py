"""Hybrid retrieval primitive (app/retrieval/hybrid.py, pseudocode 12.C)."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from app.core.events import EventType, RetrievalTrigger
from app.models.retrieval_event import RetrievalMode, RetrievalRequest
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse_bm25 import SparseIndexRegistry
from scripts.ingest_corpus import ingest_corpus
from tests.fakes import FakeEmbedder, RecordingSink


def request(
    text: str = "cancellation policy booking", corpus_id: str = "alpha", **kw
) -> RetrievalRequest:
    fields = {
        "session_id": "sess_1",
        "trace_id": "trc_1",
        "sub_query_id": "sq_1",
        "text": text,
        "corpus_id": corpus_id,
        "trigger": RetrievalTrigger.PROVISIONAL,
        "t_offset_ms": 800,
    }
    fields.update(kw)
    return RetrievalRequest(**fields)


@pytest.fixture
async def retriever_parts(test_settings, qdrant_memory):
    embedder = FakeEmbedder()
    await ingest_corpus(
        "alpha", settings=test_settings, embedder=embedder, qdrant_client=qdrant_memory
    )
    dense = DenseIndex(qdrant_memory, embedder, test_settings)
    sparse = SparseIndexRegistry(test_settings)
    return dense, sparse, test_settings


async def test_hybrid_returns_dense_and_sparse_for_the_request_corpus(retriever_parts):
    dense, sparse, settings = retriever_parts
    result = await HybridRetriever(dense, sparse, settings).retrieve(request())

    assert result.sub_query_id == "sq_1" and result.corpus_id == "alpha"
    assert result.dense and result.sparse
    assert result.dense_error is None and result.sparse_error is None
    assert {h.corpus_id for h in result.dense + result.sparse} == {"alpha"}
    assert len(result.dense) <= settings.k_dense and len(result.sparse) <= settings.k_sparse
    assert "Cancellation Policy" in result.sparse[0].text


async def test_hybrid_emits_started_and_completed_per_mode(retriever_parts):
    dense, sparse, settings = retriever_parts
    sink = RecordingSink()
    result = await HybridRetriever(dense, sparse, settings, sink=sink).retrieve(request())

    started, completed = (
        sink.of_type(EventType.RETRIEVAL_STARTED),
        sink.of_type(EventType.RETRIEVAL_COMPLETED),
    )
    assert {e.payload["mode"] for e in started} == {"dense", "sparse"}
    assert {e.payload["mode"] for e in completed} == {"dense", "sparse"}
    assert not sink.of_type(EventType.ERROR)
    for event in sink.events:
        assert event.session_id == "sess_1" and event.trace_id == "trc_1"
        assert event.payload["sub_query_id"] == "sq_1"
        assert event.payload["t_offset_ms"] == 800 and event.payload["trigger"] == "provisional"

    by_mode = {e.payload["mode"]: e.payload for e in completed}
    assert by_mode["sparse"]["result_count"] == len(result.sparse)
    assert by_mode["sparse"]["result_chunk_ids"] == [h.chunk_id for h in result.sparse]
    assert by_mode["dense"]["scores"] == [h.score for h in result.dense]
    assert all(p["latency_ms"] >= 0 for p in by_mode.values())


async def test_hybrid_returns_retrieval_event_records(retriever_parts):
    dense, sparse, settings = retriever_parts
    result = await HybridRetriever(dense, sparse, settings).retrieve(
        request(trigger=RetrievalTrigger.MULTI_INTENT)
    )
    modes = {e.mode: e for e in result.retrieval_events}
    assert set(modes) == {RetrievalMode.DENSE, RetrievalMode.SPARSE}
    assert modes[RetrievalMode.DENSE].result_chunk_ids == [h.chunk_id for h in result.dense]
    assert all(e.trigger == RetrievalTrigger.MULTI_INTENT for e in result.retrieval_events)


async def test_dense_failure_degrades_to_sparse_only(retriever_parts, monkeypatch):
    dense, sparse, settings = retriever_parts

    async def broken(*args, **kwargs):
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(dense, "search", broken)
    sink = RecordingSink()
    result = await HybridRetriever(dense, sparse, settings, sink=sink).retrieve(request())

    assert result.dense == [] and "ConnectionError" in result.dense_error
    assert result.sparse and result.sparse_error is None
    (error,) = sink.of_type(EventType.ERROR)
    assert error.payload["stage"] == "retrieval" and error.payload["recoverable"] is True
    assert error.payload["error_type"] == "ConnectionError"
    assert "dense retrieval failed for sub_query sq_1" in error.payload["message"]


async def test_missing_sparse_index_is_recorded_not_raised(retriever_parts, qdrant_memory):
    dense, _, settings = retriever_parts
    sink = RecordingSink()
    empty_registry = SparseIndexRegistry(settings, processed_dir=settings.processed_dir + "_none")
    result = await HybridRetriever(dense, empty_registry, settings, sink=sink).retrieve(request())

    assert result.sparse == [] and "SparseIndexNotFoundError" in result.sparse_error
    assert result.dense
    assert sink.of_type(EventType.ERROR)[0].payload["error_type"] == "SparseIndexNotFoundError"


async def test_timeout_contributes_no_evidence(retriever_parts, monkeypatch):
    dense, sparse, settings = retriever_parts

    async def hanging(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(sparse, "search", hanging)
    sink = RecordingSink()
    result = await HybridRetriever(dense, sparse, settings, sink=sink).retrieve(request())
    assert result.sparse == [] and result.sparse_error == "timeout"
    assert sink.of_type(EventType.ERROR)[0].payload["error_type"] == "RetrievalTimeout"


async def test_empty_query_returns_empty_results_without_errors(retriever_parts):
    dense, sparse, settings = retriever_parts
    result = await HybridRetriever(dense, sparse, settings).retrieve(request(text="   "))
    assert result.dense == [] and result.sparse == []
    assert result.dense_error is None and result.sparse_error is None


async def test_dense_and_sparse_run_concurrently(retriever_parts, monkeypatch):
    dense, sparse, settings = retriever_parts

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.2)
        return []

    monkeypatch.setattr(dense, "search", slow)
    monkeypatch.setattr(sparse, "search", slow)
    started = time.perf_counter()
    await HybridRetriever(dense, sparse, settings).retrieve(request())
    assert time.perf_counter() - started < 0.35  # ~max(0.2, 0.2), not the 0.4 sum


async def test_semaphore_bounds_concurrent_retrievals(retriever_parts, monkeypatch):
    dense, sparse, settings = retriever_parts
    limited = settings.model_copy(update={"max_concurrent_retrievals": 1})
    active = peak = 0

    async def tracked(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1
        return []

    monkeypatch.setattr(dense, "search", tracked)
    monkeypatch.setattr(sparse, "search", tracked)
    retriever = HybridRetriever(dense, sparse, limited)
    await asyncio.gather(*(retriever.retrieve(request(sub_query_id=f"sq_{i}")) for i in range(3)))
    assert peak == 2  # one sub-query at a time, each running its two modes concurrently


async def test_failing_sink_never_breaks_retrieval(retriever_parts):
    dense, sparse, settings = retriever_parts

    def exploding_sink(event):
        raise RuntimeError("telemetry backend down")

    result = await HybridRetriever(dense, sparse, settings, sink=exploding_sink).retrieve(request())
    assert result.dense and result.sparse


def test_request_rejects_unsafe_corpus_id():
    with pytest.raises(ValidationError):
        request(corpus_id="../beta")
