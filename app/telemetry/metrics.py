"""Benchmark metric formulas (docs/TELEMETRY.md §5; REQ-EVAL-01).

Every formula is a pure function computed only from a `TelemetryEvent` log plus gold labels
supplied by the caller — never by re-parsing generated text, and never by branching on a specific
benchmark scenario. This module knows the ten formulas' shapes; it holds no benchmark corpus
content, no scenario identifiers, and no gold data of its own (`benchmarks/harness.py` owns that
and calls these functions with its own gold labels as plain arguments).

Gate thresholds and pass/fail mapping live in docs/EVALUATION.md §1, not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.core.events import EventType, TelemetryEvent


def _first_timestamp(events: Sequence[TelemetryEvent], event_type: EventType) -> datetime | None:
    for event in events:
        if event.event_type == event_type:
            return event.timestamp
    return None


# --- G2: Early Retrieval Rate ------------------------------------------------------------------


@dataclass(frozen=True)
class EarlyRetrievalCase:
    """One held-out streaming prompt's early-retrieval eligibility (gold) and outcome (observed).

    `eligible` is False for a gold NO_RETRIEVAL case (TELEMETRY.md §5's "eligible excludes cases
    with gold label NO_RETRIEVAL") - such a case never enters the rate's denominator.
    """

    eligible: bool
    utterance_end_offset_ms: int
    first_retrieval_started_offset_ms: int | None  # None if retrieval never started


def early_retrieval_rate(cases: Iterable[EarlyRetrievalCase]) -> float | None:
    eligible = [c for c in cases if c.eligible]
    if not eligible:
        return None
    early = sum(
        1
        for c in eligible
        if c.first_retrieval_started_offset_ms is not None
        and c.first_retrieval_started_offset_ms < c.utterance_end_offset_ms
    )
    return early / len(eligible)


# --- G3: Multi-Intent Accuracy ------------------------------------------------------------------


def multi_intent_accuracy(
    cases: Iterable[tuple[Sequence[str], Sequence[str]]],
) -> float | None:
    """Each case is (predicted_sub_intents, gold_sub_intents); a case counts as correct only on an
    exact set match (TELEMETRY.md §5: "set match")."""
    cases = list(cases)
    if not cases:
        return None
    correct = sum(1 for predicted, gold in cases if set(predicted) == set(gold))
    return correct / len(cases)


# --- G4: Citation Grounding Rate / Fabricated Citation Rate -------------------------------------


def citation_grounding_rate(*, grounded_assertions: int, sampled_assertions: int) -> float | None:
    if sampled_assertions == 0:
        return None
    return grounded_assertions / sampled_assertions


def fabricated_citation_rate(
    citation_events: Sequence[TelemetryEvent], evidence_chunk_ids: Sequence[str]
) -> float | None:
    """A citation is fabricated iff its chunk_id is absent from that turn's own evidence set
    (TELEMETRY.md §5) - the gate requires this to equal 0. `citation_events` should already be
    scoped to the one trace/turn whose `evidence_chunk_ids` set is passed in."""
    if not citation_events:
        return None
    valid_ids = set(evidence_chunk_ids)
    fabricated = sum(1 for e in citation_events if e.payload.get("chunk_id") not in valid_ids)
    return fabricated / len(citation_events)


# --- Retrieval Precision / Recall ---------------------------------------------------------------


def retrieval_precision(
    retrieved_chunk_ids: Sequence[str], gold_relevant_chunk_ids: Sequence[str]
) -> float | None:
    if not retrieved_chunk_ids:
        return None
    gold = set(gold_relevant_chunk_ids)
    relevant_retrieved = sum(1 for c in retrieved_chunk_ids if c in gold)
    return relevant_retrieved / len(retrieved_chunk_ids)


def retrieval_recall(
    retrieved_chunk_ids: Sequence[str], gold_relevant_chunk_ids: Sequence[str]
) -> float | None:
    gold = set(gold_relevant_chunk_ids)
    if not gold:
        return None
    retrieved = set(retrieved_chunk_ids)
    relevant_retrieved = sum(1 for c in gold if c in retrieved)
    return relevant_retrieved / len(gold)


# --- Latency ------------------------------------------------------------------------------------


def time_to_first_token(events: Sequence[TelemetryEvent], *, turn_start: datetime) -> float | None:
    """TTFT = timestamp(first ANSWER_DELTA) - timestamp(RETRIEVE decision OR utterance_end,
    whichever applies) - `turn_start` is the caller's choice of that reference point, in seconds.
    """
    first_delta = _first_timestamp(events, EventType.ANSWER_DELTA)
    if first_delta is None:
        return None
    return (first_delta - turn_start).total_seconds()


def end_to_end_latency(
    events: Sequence[TelemetryEvent], *, utterance_end: datetime
) -> float | None:
    completed = _first_timestamp(events, EventType.ANSWER_VERSION_CREATED)
    if completed is None:
        return None
    return (completed - utterance_end).total_seconds()


def token_cost(*, prompt_tokens: int, completion_tokens: int, provider_rate: float) -> float:
    """provider_rate is cost per token (already combining prompt/completion pricing at the
    caller's chosen granularity); kept as a single multiplier so this formula never hardcodes any
    provider's pricing table."""
    return (prompt_tokens + completion_tokens) * provider_rate


# --- G5: Session Refinement Accuracy ------------------------------------------------------------


@dataclass(frozen=True)
class RefinementCase:
    carried_forward_claims_unchanged: bool
    only_delta_subqueries_issued: bool
    session_state_cleared: bool


def session_refinement_accuracy(cases: Iterable[RefinementCase]) -> float | None:
    cases = list(cases)
    if not cases:
        return None
    correct = sum(
        1
        for c in cases
        if c.carried_forward_claims_unchanged
        and c.only_delta_subqueries_issued
        and not c.session_state_cleared
    )
    return correct / len(cases)
