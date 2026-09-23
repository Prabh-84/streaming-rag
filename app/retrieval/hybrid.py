"""Hybrid retrieval primitive for one sub-query (pseudocode 12.C, REQ-EVID-01).

Runs dense (Qdrant) and sparse (BM25) search concurrently for a single sub-query, scoped to the
request's corpus_id. A failed or timed-out mode contributes no evidence and never takes the other
mode down with it. Fan-out across sub-queries and RRF fusion are Phase 5 (fusion.py); a semaphore
here bounds concurrent retrievals (MAX_CONCURRENT_RETRIEVALS) so that fan-out is protected.

Telemetry per mode: RETRIEVAL_STARTED, then RETRIEVAL_COMPLETED (carrying the RetrievalEvent
record — docs/TELEMETRY.md §2.1 names RetrievalEvent as the record backing both), plus ERROR on
failure (original TRD §9.3).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from app.core.config import Settings, get_settings
from app.core.events import EventSink, EventType, TelemetryEvent
from app.models.chunk import RetrievedChunk
from app.models.retrieval_event import (
    RetrievalEvent,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
)
from app.retrieval.dense import DenseIndex
from app.retrieval.sparse_bm25 import SparseIndexRegistry

log = structlog.get_logger()


@dataclass(frozen=True)
class _ModeOutcome:
    hits: list[RetrievedChunk]
    event: RetrievalEvent
    error: str | None


class HybridRetriever:
    def __init__(
        self,
        dense: DenseIndex,
        sparse: SparseIndexRegistry,
        settings: Settings | None = None,
        *,
        sink: EventSink | None = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._settings = settings or get_settings()
        self._sink = sink
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_retrievals)

    async def retrieve(
        self, request: RetrievalRequest, *, sink: EventSink | None = None
    ) -> RetrievalResult:
        """`sink` overrides the sink this HybridRetriever was constructed with, for this call
        only. One shared retriever instance (e.g. the process-wide app.state one) has no single
        fixed telemetry destination — each WebSocket connection routes events to its own
        session's queue, so the caller supplies that per call rather than per instance."""
        s = self._settings
        effective_sink = sink if sink is not None else self._sink
        async with self._semaphore:
            dense, sparse = await asyncio.gather(
                self._run_mode(
                    request,
                    RetrievalMode.DENSE,
                    lambda: self._dense.search(
                        request.corpus_id,
                        request.text,
                        s.k_dense,
                        timeout_ms=s.retrieval_timeout_ms,
                    ),
                    effective_sink,
                ),
                self._run_mode(
                    request,
                    RetrievalMode.SPARSE,
                    lambda: self._sparse.search(
                        request.corpus_id,
                        request.text,
                        s.k_sparse,
                        timeout_ms=s.retrieval_timeout_ms,
                    ),
                    effective_sink,
                ),
            )
        return RetrievalResult(
            sub_query_id=request.sub_query_id,
            corpus_id=request.corpus_id,
            dense=dense.hits,
            sparse=sparse.hits,
            dense_error=dense.error,
            sparse_error=sparse.error,
            retrieval_events=[dense.event, sparse.event],
        )

    async def _run_mode(
        self,
        request: RetrievalRequest,
        mode: RetrievalMode,
        search: Callable[[], Awaitable[list[RetrievedChunk]]],
        sink: EventSink | None,
    ) -> _ModeOutcome:
        self._emit(
            sink,
            request,
            EventType.RETRIEVAL_STARTED,
            {
                "sub_query_id": request.sub_query_id,
                "mode": mode.value,
                "t_offset_ms": request.t_offset_ms,
                "trigger": request.trigger.value,
            },
        )
        started = time.perf_counter()
        hits: list[RetrievedChunk] = []
        error: str | None = None
        error_type = ""
        try:
            hits = await search()
        except TimeoutError:
            error, error_type = "timeout", "RetrievalTimeout"
        except Exception as exc:
            error, error_type = f"{type(exc).__name__}: {exc}", type(exc).__name__
        latency_ms = round((time.perf_counter() - started) * 1000)

        event = RetrievalEvent(
            session_id=request.session_id,
            sub_query_id=request.sub_query_id,
            mode=mode,
            result_chunk_ids=[h.chunk_id for h in hits],
            scores=[h.score for h in hits],
            latency_ms=latency_ms,
            t_offset_ms=request.t_offset_ms,
            trigger=request.trigger,
        )
        self._emit(
            sink,
            request,
            EventType.RETRIEVAL_COMPLETED,
            {
                "sub_query_id": request.sub_query_id,
                "mode": mode.value,
                "result_count": len(hits),
                "latency_ms": latency_ms,
                "result_chunk_ids": event.result_chunk_ids,
                "scores": event.scores,
                "t_offset_ms": request.t_offset_ms,
                "trigger": request.trigger.value,
            },
        )
        if error is not None:
            self._emit(
                sink,
                request,
                EventType.ERROR,
                {
                    "stage": "retrieval",
                    "error_type": error_type,
                    "message": f"{mode.value} retrieval failed for sub_query "
                    f"{request.sub_query_id}: {error}",
                    "recoverable": True,
                },
            )
        return _ModeOutcome(hits=hits, event=event, error=error)

    def _emit(
        self,
        sink: EventSink | None,
        request: RetrievalRequest,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> None:
        if sink is None:
            return
        try:
            sink(
                TelemetryEvent(
                    session_id=request.session_id,
                    trace_id=request.trace_id,
                    event_type=event_type,
                    payload=payload,
                )
            )
        except Exception:
            # Telemetry must never break the retrieval path.
            log.exception("telemetry_sink_failed", event_type=event_type.value)
