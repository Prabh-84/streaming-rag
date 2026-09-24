"""Benchmark metric formulas (docs/TELEMETRY.md §5; app/telemetry/metrics.py).

These are pure-function unit tests against synthetic event logs and gold labels - no benchmark
suite content, no BENCH-nn identifiers, nothing that would violate REQ-EVAL-01's boundary (that
belongs to benchmarks/, never to /app or its tests).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.events import EventType, TelemetryEvent
from app.telemetry import metrics


def _event(event_type: EventType, payload: dict, *, ts: datetime | None = None) -> TelemetryEvent:
    kwargs = {
        "session_id": "sess_1",
        "trace_id": "trc_1",
        "event_type": event_type,
        "payload": payload,
    }
    if ts is not None:
        kwargs["timestamp"] = ts
    return TelemetryEvent(**kwargs)


# --- Early Retrieval Rate ------------------------------------------------------------------------


def test_early_retrieval_rate_counts_only_eligible_cases():
    cases = [
        metrics.EarlyRetrievalCase(
            eligible=True, utterance_end_offset_ms=2000, first_retrieval_started_offset_ms=500
        ),
        metrics.EarlyRetrievalCase(
            eligible=True, utterance_end_offset_ms=2000, first_retrieval_started_offset_ms=2500
        ),
        # Not eligible (gold NO_RETRIEVAL) - excluded from both numerator and denominator.
        metrics.EarlyRetrievalCase(
            eligible=False, utterance_end_offset_ms=2000, first_retrieval_started_offset_ms=None
        ),
    ]
    assert metrics.early_retrieval_rate(cases) == 0.5


def test_early_retrieval_rate_none_when_no_eligible_cases():
    assert metrics.early_retrieval_rate([]) is None


def test_early_retrieval_rate_never_started_counts_as_not_early():
    cases = [
        metrics.EarlyRetrievalCase(
            eligible=True, utterance_end_offset_ms=1000, first_retrieval_started_offset_ms=None
        )
    ]
    assert metrics.early_retrieval_rate(cases) == 0.0


# --- Multi-Intent Accuracy -----------------------------------------------------------------------


def test_multi_intent_accuracy_requires_exact_set_match():
    cases = [
        (["cancellation", "catering"], ["cancellation", "catering"]),  # match
        (["cancellation"], ["cancellation", "catering"]),  # under-decomposed
        (["cancellation", "catering", "parking"], ["cancellation", "catering"]),  # over
    ]
    assert metrics.multi_intent_accuracy(cases) == 1 / 3


def test_multi_intent_accuracy_ignores_order():
    cases = [(["a", "b"], ["b", "a"])]
    assert metrics.multi_intent_accuracy(cases) == 1.0


# --- Citation Grounding / Fabrication ---------------------------------------------------------


def test_citation_grounding_rate_basic_ratio():
    assert metrics.citation_grounding_rate(grounded_assertions=17, sampled_assertions=20) == 0.85


def test_citation_grounding_rate_none_when_nothing_sampled():
    assert metrics.citation_grounding_rate(grounded_assertions=0, sampled_assertions=0) is None


def test_fabricated_citation_rate_flags_ids_outside_the_evidence_set():
    events = [
        _event(EventType.CITATION_CREATED, {"chunk_id": "c1"}),
        _event(EventType.CITATION_CREATED, {"chunk_id": "c2"}),
        _event(EventType.CITATION_CREATED, {"chunk_id": "not_in_evidence"}),
    ]
    rate = metrics.fabricated_citation_rate(events, evidence_chunk_ids=["c1", "c2", "c3"])
    assert rate == 1 / 3


def test_fabricated_citation_rate_zero_when_every_citation_is_grounded():
    events = [_event(EventType.CITATION_CREATED, {"chunk_id": "c1"})]
    assert metrics.fabricated_citation_rate(events, evidence_chunk_ids=["c1"]) == 0.0


# --- Retrieval Precision / Recall -----------------------------------------------------------------


def test_retrieval_precision_and_recall():
    retrieved = ["c1", "c2", "c3", "c4"]
    gold = ["c1", "c3", "c5"]
    assert metrics.retrieval_precision(retrieved, gold) == 0.5  # 2 of 4 retrieved are relevant
    assert metrics.retrieval_recall(retrieved, gold) == 2 / 3  # 2 of 3 gold chunks retrieved


def test_retrieval_precision_none_when_nothing_retrieved():
    assert metrics.retrieval_precision([], ["c1"]) is None


def test_retrieval_recall_none_when_no_gold_relevant_chunks():
    assert metrics.retrieval_recall(["c1"], []) is None


# --- Latency --------------------------------------------------------------------------------------


def test_time_to_first_token_measured_from_turn_start():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = [_event(EventType.ANSWER_DELTA, {}, ts=start + timedelta(seconds=1.5))]
    assert metrics.time_to_first_token(events, turn_start=start) == 1.5


def test_time_to_first_token_none_without_an_answer_delta():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert metrics.time_to_first_token([], turn_start=start) is None


def test_end_to_end_latency_measured_from_utterance_end():
    end = datetime(2026, 1, 1, tzinfo=UTC)
    events = [_event(EventType.ANSWER_VERSION_CREATED, {}, ts=end + timedelta(seconds=3.0))]
    assert metrics.end_to_end_latency(events, utterance_end=end) == 3.0


def test_token_cost_multiplies_total_tokens_by_rate():
    assert metrics.token_cost(prompt_tokens=100, completion_tokens=50, provider_rate=0.001) == 0.15


# --- Session Refinement Accuracy -----------------------------------------------------------------


def test_session_refinement_accuracy_requires_all_three_conditions():
    cases = [
        metrics.RefinementCase(
            carried_forward_claims_unchanged=True,
            only_delta_subqueries_issued=True,
            session_state_cleared=False,
        ),
        metrics.RefinementCase(
            carried_forward_claims_unchanged=True,
            only_delta_subqueries_issued=False,  # re-decomposed the whole buffer - fails
            session_state_cleared=False,
        ),
    ]
    assert metrics.session_refinement_accuracy(cases) == 0.5


def test_session_refinement_accuracy_none_when_no_cases():
    assert metrics.session_refinement_accuracy([]) is None
