"""Focused tests for the Phase 9 evaluation harness itself (benchmarks/harness.py).

These test the harness's own mechanics - grading logic, gate aggregation, and that a real replay
through the actual WS/HTTP API produces a gradeable trace - using `use_fakes=True` (deterministic,
offline) so this suite never depends on a real LLM call or network access. The one real,
non-fake, fully-measured run (whatever LLM_PROVIDER is configured) is `scripts/run_benchmark.py`,
run manually to report genuine metrics - never part of the automated test suite.
"""

from __future__ import annotations

from app.core.events import EventType, TelemetryEvent
from benchmarks import harness


def _event(event_type: EventType, payload: dict, *, trace_id: str = "trc_1") -> TelemetryEvent:
    return TelemetryEvent(
        session_id="sess_1", trace_id=trace_id, event_type=event_type, payload=payload
    )


# --- 1. Grading logic (synthetic event logs - no real replay needed) ----------------------------


def test_grade_scenario_passes_when_decision_sequence_matches_gold():
    scenario = {
        "id": "X",
        "description": "d",
        "gold": {"decisions": ["WAIT", "RETRIEVE"]},
    }
    events = [
        _event(EventType.RETRIEVAL_DECISION, {"decision": "WAIT", "trigger": None, "reason": "x"}),
        _event(
            EventType.RETRIEVAL_DECISION,
            {"decision": "RETRIEVE", "trigger": "final", "reason": "final"},
        ),
    ]
    result = harness._grade_scenario(scenario, events)
    assert result.passed
    assert result.failures == []


def test_grade_scenario_fails_on_decision_sequence_mismatch():
    scenario = {"id": "X", "description": "d", "gold": {"decisions": ["WAIT", "WAIT"]}}
    events = [
        _event(EventType.RETRIEVAL_DECISION, {"decision": "WAIT", "trigger": None, "reason": "x"}),
        _event(
            EventType.RETRIEVAL_DECISION,
            {"decision": "RETRIEVE", "trigger": "final", "reason": "final"},
        ),
    ]
    result = harness._grade_scenario(scenario, events)
    assert not result.passed
    assert any("decision sequence" in f for f in result.failures)


def test_grade_scenario_flags_too_many_sub_queries():
    scenario = {"id": "X", "description": "d", "gold": {"max_sub_queries": 1}}
    events = [
        _event(EventType.SUBQUERY_CREATED, {"sub_query_id": "a", "text": "a", "intent_label": "a"}),
        _event(EventType.SUBQUERY_CREATED, {"sub_query_id": "b", "text": "b", "intent_label": "b"}),
    ]
    result = harness._grade_scenario(scenario, events)
    assert not result.passed


def test_grade_scenario_expect_uncertainty_fails_when_absent():
    scenario = {"id": "X", "description": "d", "gold": {"expect_uncertainty": True}}
    result = harness._grade_scenario(scenario, [])
    assert not result.passed


def test_grade_scenario_no_sequence_gap_fails_on_a_gap_error():
    scenario = {"id": "X", "description": "d", "gold": {"no_sequence_gap": True}}
    events = [
        _event(
            EventType.ERROR,
            {
                "stage": "stream",
                "error_type": "SequenceGap",
                "message": "sequence_gap",
                "recoverable": True,
            },
        )
    ]
    result = harness._grade_scenario(scenario, events)
    assert not result.passed


def test_grade_scenario_expect_contradiction_pair_requires_co_occurrence():
    scenario = {
        "id": "X",
        "description": "d",
        "gold": {"expect_contradiction_pair": ["c1", "c2"]},
    }
    only_one = [
        _event(
            EventType.RERANK_COMPLETED,
            {"sub_query_id": "s", "ranked_chunk_ids": ["c1"], "scores": [0.5]},
        )
    ]
    assert not harness._grade_scenario(scenario, only_one).passed

    both = [
        _event(
            EventType.RERANK_COMPLETED,
            {"sub_query_id": "s", "ranked_chunk_ids": ["c1", "c2"], "scores": [0.5, 0.4]},
        )
    ]
    assert harness._grade_scenario(scenario, both).passed


def test_grade_bench10_fails_if_a_fabricated_citation_reaches_the_event_log():
    scenario = {
        "id": "BENCH-10",
        "description": "d",
        "gold": {"expect_zero_fabricated_citations": True},
    }
    events = [
        _event(
            EventType.CITATION_CREATED,
            {"chunk_id": "bogus", "doc_id": "nonexistent_doc", "section": "§99", "claim_text": "x"},
        )
    ]
    result = harness._grade_bench10(scenario, events)
    assert not result.passed


def test_grade_bench10_passes_when_no_fabricated_citation_is_emitted():
    scenario = {
        "id": "BENCH-10",
        "description": "d",
        "gold": {"expect_zero_fabricated_citations": True},
    }
    events = [
        _event(
            EventType.UNCERTAINTY,
            {"sub_query_id": "s", "reason": "missing_citation", "clarifying_question": None},
        )
    ]
    result = harness._grade_bench10(scenario, events)
    assert result.passed


# --- 2. Scenario data integrity -------------------------------------------------------------------


def test_scenarios_file_defines_all_ten_bench_cases_with_required_fields():
    suite = harness.load_scenarios()
    ids = [s["id"] for s in suite["scenarios"]]
    assert ids == [f"BENCH-{i:02d}" for i in range(1, 11)]
    for scenario in suite["scenarios"]:
        assert scenario["chunks"], f"{scenario['id']} has no chunks"
        assert "gold" in scenario and scenario["gold"], f"{scenario['id']} has no gold labels"


# --- 3. A real replay produces a gradeable trace (offline, deterministic) -----------------------


def test_real_replay_of_a_simple_scenario_produces_a_complete_gradeable_trace(tmp_path):
    """End-to-end proof the harness's own plumbing (ingest -> WS replay -> GET /events -> grade)
    works against the real app/WS/HTTP stack, using use_fakes=True so this stays fast and
    network-free. This is not "the documented benchmark suite" (that is scripts/run_benchmark.py,
    run for real, separately) - just this module's own mechanics."""
    suite = harness.load_scenarios()
    scenario = next(s for s in suite["scenarios"] if s["id"] == "BENCH-01")
    result = harness._run_scenario(scenario, tmp_path, corpus_id=suite["corpus_id"], use_fakes=True)
    assert result.passed, result.failures
    assert any(e.event_type == EventType.RETRIEVAL_DECISION for e in result.events)
    assert any(e.event_type == EventType.CITATION_CREATED for e in result.events)
    # every event this trace produced belongs to the one session it was replayed into.
    assert all(e.session_id == result.events[0].session_id for e in result.events)


# --- 4. Gate aggregation -------------------------------------------------------------------------


def test_aggregate_gates_computes_g2_only_from_eligible_scenarios():
    scenarios = [
        {
            "id": "A",
            "gold": {"eligible_for_early_retrieval": True, "utterance_end_offset_ms": 1000},
        },
        {"id": "B", "gold": {}},  # no early-retrieval opinion - excluded
    ]
    results = {
        "A": harness.ScenarioResult(
            scenario_id="A",
            description="d",
            passed=True,
            failures=[],
            events=[
                _event(
                    EventType.RETRIEVAL_STARTED,
                    {
                        "sub_query_id": "s",
                        "mode": "dense",
                        "t_offset_ms": 100,
                        "trigger": "provisional",
                    },
                )
            ],
        ),
        "B": harness.ScenarioResult(
            scenario_id="B", description="d", passed=True, failures=[], events=[]
        ),
    }
    gates, metrics = harness._aggregate_gates(scenarios, results)
    assert metrics["early_retrieval_rate"] == 1.0
    assert gates["G2"]["pass"] is True


def test_aggregate_gates_g4_fabrication_flags_any_fabricated_citation():
    scenarios = [{"id": "A", "gold": {}}]
    events = [
        _event(EventType.CITATION_CREATED, {"chunk_id": "c1", "doc_id": "real_doc"}),
        _event(EventType.CITATION_CREATED, {"chunk_id": "c2", "doc_id": "nonexistent_doc"}),
    ]
    results = {
        "A": harness.ScenarioResult(
            scenario_id="A", description="d", passed=True, failures=[], events=events
        )
    }
    gates, metrics = harness._aggregate_gates(scenarios, results)
    assert metrics["fabricated_citation_rate"] == 0.5
    assert gates["G4_fabrication"]["pass"] is False
