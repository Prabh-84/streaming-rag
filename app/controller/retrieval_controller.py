"""Retrieval Controller: WAIT / RETRIEVE / NO_RETRIEVAL per chunk (pseudocode 12.A;
REQ-CTRL-01/02/03, REQ-STREAM-01/02, REQ-SUPPRESS-01/02).

Retrieval never fires from stability or an entity alone — only their conjunction (REQ-CTRL-01/02),
which is exactly what pitfall 1 (eager/permanent retrieval) targets.
`session.has_retrieved_for_topic` then prevents re-firing on every later chunk of the same stable
topic (REQ-CTRL-01's "no_new_entity" branch) — only a chunk that changes `entities` re-triggers.

On RETRIEVE, the buffer is handed to the Multi-Intent Decomposer (REQ-INTENT-01/02) before
retrieval — a downstream refinement of an already-made decision, never a second retrieval trigger
(original TRD §6.2). A non-compound request decomposes to exactly one sub-query, so this is a
behavior-preserving extension of Phase 3: single-intent requests still issue exactly one retrieval
call, just now routed through the decomposer's trivial fallback path.

Out of scope for this phase (later phases): RRF/evidence fusion, cross-encoder reranking,
session-answer refinement, grounding, answer generation. `check_suppression`'s ambiguous-match
branch is deterministic-only here (no LLM call) — see app.controller.suppression.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from app.controller.entity_extraction import entity_diff, extract_entities
from app.controller.suppression import check_suppression
from app.core.config import Settings, get_settings
from app.core.embeddings import Embedder, get_embedder
from app.core.events import EventSink, EventType, RetrievalTrigger, TelemetryEvent
from app.decomposition.multi_intent import DecompositionLLM, decompose
from app.models.retrieval_event import RetrievalRequest, RetrievalResult
from app.models.transcript_chunk import TranscriptChunk
from app.retrieval.hybrid import HybridRetriever
from app.session.session_store import SessionRecord

WAIT = "WAIT"
RETRIEVE = "RETRIEVE"
NO_RETRIEVAL = "NO_RETRIEVAL"


@dataclass(frozen=True)
class ControllerDecision:
    decision: str  # WAIT | RETRIEVE | NO_RETRIEVAL
    reason: str
    trigger: RetrievalTrigger | None = None
    sub_query_ids: list[str] = field(default_factory=list)
    retrievals: list[RetrievalResult] = field(default_factory=list)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def compute_stability(history: list[tuple[float, ...]], window: int) -> float:
    """Average pairwise cosine similarity among the last `window` embeddings. At window=2 (the
    frozen default) this is exactly the "inter-chunk cosine similarity" REQ-CTRL-01 names and the
    state-machine transition guard's "last 2 chunks" (PRD_TRD.md §5.4). Fewer than 2 embeddings
    (the session's first chunk) can't be judged stable, so this returns 0.0."""
    recent = history[-window:]
    if len(recent) < 2:
        return 0.0
    pairs = [(recent[i], recent[j]) for i in range(len(recent)) for j in range(i + 1, len(recent))]
    return sum(_cosine(a, b) for a, b in pairs) / len(pairs)


async def on_chunk(
    session: SessionRecord,
    chunk: TranscriptChunk,
    *,
    trace_id: str,
    retriever: HybridRetriever,
    embedder: Embedder | None = None,
    llm: DecompositionLLM | None = None,
    settings: Settings | None = None,
    sink: EventSink | None = None,
) -> ControllerDecision:
    """One pass of pseudocode 12.A for one already-order-committed chunk. Callers run this once
    per chunk returned by SessionRecord.ingest_chunk(), in order."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder()

    embedding = await asyncio.to_thread(embedder.embed, chunk.text_delta)
    session.embedding_history.append(embedding)

    buffer_text = session.buffer_text()

    suppression = await asyncio.to_thread(
        check_suppression,
        buffer_text,
        session.entities,
        session.corpus_id,
        has_prior_answer=bool(session.answer_versions),
    )
    if suppression is not None:
        # REQ-CTRL-03's literal reason vocabulary for the controller's own decision; the more
        # specific "presentation_restructure" label is preserved on the SuppressionResult itself
        # for whichever future Synthesis stage consumes it.
        return _decide(session, trace_id, sink, NO_RETRIEVAL, "presentation_only")

    new_entities = await asyncio.to_thread(extract_entities, buffer_text, session.corpus_id)
    delta = entity_diff(new_entities, session.entities)
    session.entities.update(new_entities)

    if not delta and session.has_retrieved_for_topic:
        return _decide(session, trace_id, sink, WAIT, "no_new_entity")

    stability = compute_stability(session.embedding_history, settings.stability_window)
    if stability < settings.stability_threshold or not delta:
        session.wait_count += 1
        if session.wait_count > settings.max_wait_chunks:
            session.wait_count = 0
            return await _retrieve(
                session,
                chunk,
                trace_id,
                retriever,
                sink,
                RetrievalTrigger.FORCED_AFTER_MAX_WAIT,
                llm=llm,
                settings=settings,
            )
        return _decide(session, trace_id, sink, WAIT, "intent_unstable")

    session.wait_count = 0
    session.has_retrieved_for_topic = True
    trigger = RetrievalTrigger.FINAL if chunk.is_final else RetrievalTrigger.PROVISIONAL
    return await _retrieve(
        session, chunk, trace_id, retriever, sink, trigger, llm=llm, settings=settings
    )


async def _retrieve(
    session: SessionRecord,
    chunk: TranscriptChunk,
    trace_id: str,
    retriever: HybridRetriever,
    sink: EventSink | None,
    trigger: RetrievalTrigger,
    *,
    llm: DecompositionLLM | None = None,
    settings: Settings | None = None,
) -> ControllerDecision:
    settings = settings or get_settings()
    _emit_event(session, trace_id, sink, RETRIEVE, trigger.value, trigger)

    sub_queries = await decompose(
        session.buffer_text(),
        session.corpus_id,
        session.session_id,
        chunk.t_offset_ms,
        trace_id=trace_id,
        llm=llm,
        settings=settings,
        sink=sink,
    )

    requests = [
        RetrievalRequest(
            session_id=session.session_id,
            trace_id=trace_id,
            sub_query_id=sub_query.sub_query_id,
            text=sub_query.text,
            corpus_id=session.corpus_id,
            # The first sub-query keeps the controller's own trigger; any additional sub-query
            # produced by genuine decomposition is tagged multi_intent (REQ-OBS-05), distinct
            # from the trigger that caused the decision to retrieve in the first place.
            trigger=trigger if i == 0 else RetrievalTrigger.MULTI_INTENT,
            t_offset_ms=chunk.t_offset_ms,
        )
        for i, sub_query in enumerate(sub_queries)
    ]
    results = await asyncio.gather(
        *(retriever.retrieve(request, sink=sink) for request in requests)
    )

    return ControllerDecision(
        decision=RETRIEVE,
        reason=trigger.value,
        trigger=trigger,
        sub_query_ids=[sq.sub_query_id for sq in sub_queries],
        retrievals=list(results),
    )


def _decide(
    session: SessionRecord,
    trace_id: str,
    sink: EventSink | None,
    decision: str,
    reason: str,
) -> ControllerDecision:
    _emit_event(session, trace_id, sink, decision, reason, None)
    return ControllerDecision(decision=decision, reason=reason)


def _emit_event(
    session: SessionRecord,
    trace_id: str,
    sink: EventSink | None,
    decision: str,
    reason: str,
    trigger: RetrievalTrigger | None,
) -> None:
    if sink is None:
        return
    sink(
        TelemetryEvent(
            session_id=session.session_id,
            trace_id=trace_id,
            event_type=EventType.RETRIEVAL_DECISION,
            payload={
                "decision": decision,
                "trigger": trigger.value if trigger else None,
                "reason": reason,
            },
        )
    )
