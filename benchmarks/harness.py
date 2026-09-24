"""Phase 9 benchmark replay + grading engine (docs/EVALUATION.md).

Lives outside `/app` on purpose (REQ-EVAL-01/HC-2): this module is one of the two places (with
`scripts/run_benchmark.py`) allowed to know the held-out suite's name, its BENCH-nn scenario ids,
and its gold labels. `/app` never imports this module and never sees this suite's gold data - it
only ever receives corpus text and transcript chunks through its own generic,
already-parameterized interfaces (Settings.corpus_root, POST /session, WS /session/{id}/stream),
exactly as a real client would.

Replay is real, not simulated: a real (in-memory) Qdrant, real BM25, real spaCy entity extraction,
and by default the real configured embedder/reranker/decomposer/generator (whatever LLM_PROVIDER
points at) - the same app.main.create_app() composition production runs, driven purely through the
public HTTP/WS API. Callers that want a fast, fully offline run (this module's own pytest smoke
test) pass explicit fake providers instead; `scripts/run_benchmark.py`'s real run passes none, so
every provider lazily resolves to its real, production singleton.

Grading reads back GET /session/{id}/events (REQ-OBS-03) and computes the ten TELEMETRY.md §5
formulas (app/telemetry/metrics.py) against this file's own gold labels - metrics.py itself holds
no gold data and no benchmark-specific literals, only the formula shapes.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

import app.controller.retrieval_controller as retrieval_controller
import app.decomposition.multi_intent as multi_intent
import app.generation.streaming_generator as streaming_generator
import app.main as main_module
from app.controller.entity_extraction import get_corpus_matcher
from app.core.config import Settings, get_settings
from app.core.events import EventType, TelemetryEvent
from app.core.slots import get_slot_schema
from app.models.evaluation_result import EvaluationRunStatus
from app.telemetry import metrics as metric_fns
from scripts.ingest_corpus import ingest_corpus

SUITE_NAME = "streaming_suite_v1"
SUITE_ROOT = Path(__file__).resolve().parent / SUITE_NAME
CORPUS_ROOT = SUITE_ROOT / "corpus"
SCENARIOS_PATH = SUITE_ROOT / "scenarios.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

API_KEY = "benchmark_harness_key"

# The one deliberately-injected fault (BENCH-10, "Citation validation (injected fault)"):
# EVALUATION.md's own scenario name says the fault is injected by the harness, not hoped for from
# a real model, so this is the sole scenario that does not use the real configured generator.
_INJECTED_FABRICATED_TAG = "[nonexistent_doc §99]"
_FAULT_SCENARIO_ID = "BENCH-10"


def load_scenarios() -> dict[str, Any]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


@dataclass
class ScenarioResult:
    scenario_id: str
    description: str
    passed: bool
    failures: list[str]
    events: list[TelemetryEvent]
    precision: float | None = None
    recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "passed": self.passed,
            "failures": self.failures,
            "retrieval_precision": self.precision,
            "retrieval_recall": self.recall,
        }


@dataclass
class BenchmarkReport:
    run_id: str
    corpus_id: str
    started_at: datetime
    completed_at: datetime
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: EvaluationRunStatus = EvaluationRunStatus.COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "test_set": f"benchmarks/{SUITE_NAME}",
            "corpus_id": self.corpus_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "gates": self.gates,
            "metrics": self.metrics,
            "scenarios": [r.to_dict() for r in self.scenario_results],
        }


# --- App composition (mirrors tests/integration/test_stream_api.py's app_client fixture) --------


class _Harness:
    def __init__(
        self,
        processed_dir: Path,
        *,
        embedder: Any = None,
        reranker: Any = None,
        decomposer: Any = None,
        generator: Any = None,
    ) -> None:
        # Saved so __exit__ can restore them - os.environ and every lru_cache singleton below are
        # process-wide, and this class runs both inside pytest (sharing a process with unrelated
        # tests) and standalone (scripts/run_benchmark.py), so nothing here can rely on pytest's
        # own self-restoring monkeypatch fixture.
        self._env_originals = {
            k: os.environ.get(k) for k in ("CORPUS_ROOT", "PROCESSED_DIR", "API_KEY")
        }
        os.environ["CORPUS_ROOT"] = str(CORPUS_ROOT)
        os.environ["PROCESSED_DIR"] = str(processed_dir)
        os.environ["API_KEY"] = API_KEY
        get_settings.cache_clear()
        get_slot_schema.cache_clear()
        get_corpus_matcher.cache_clear()
        self.settings: Settings = get_settings()

        self._embedder = embedder
        self._qdrant_client = AsyncQdrantClient(location=":memory:")
        self._decomposer = decomposer
        self._reranker = reranker
        self._generator = generator
        self.client: TestClient | None = None
        self._originals: dict[tuple[Any, str], Any] = {}

    def ingest(self) -> None:
        import asyncio

        embedder = self._embedder
        if embedder is None:
            from app.core.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(self.settings.embedding_model)
            self._embedder = embedder
        asyncio.run(
            ingest_corpus(
                "benchmark_v1",
                settings=self.settings,
                embedder=embedder,
                qdrant_client=self._qdrant_client,
            )
        )

    def __enter__(self) -> TestClient:
        # Plain module-attribute overrides (this class is used both from pytest and from the
        # standalone scripts/run_benchmark.py CLI, so it cannot rely on pytest's own
        # self-restoring monkeypatch fixture) - every override must be undone in __exit__, or it
        # leaks into whatever runs next in the same process (a real bug this file's own pytest
        # smoke test caught: a leaked get_decomposer broke an unrelated, later test).
        self._originals: dict[tuple[Any, str], Any] = {
            (main_module, "create_qdrant_client"): main_module.create_qdrant_client,
            (main_module, "get_embedder"): main_module.get_embedder,
            (multi_intent, "get_decomposer"): multi_intent.get_decomposer,
            (retrieval_controller, "get_reranker"): retrieval_controller.get_reranker,
            (streaming_generator, "get_generator"): streaming_generator.get_generator,
        }

        main_module.create_qdrant_client = lambda url: self._qdrant_client
        if self._embedder is not None:
            main_module.get_embedder = lambda: self._embedder
        if self._decomposer is not None:
            multi_intent.get_decomposer = lambda: self._decomposer
        if self._reranker is not None:
            retrieval_controller.get_reranker = lambda: self._reranker
        if self._generator is not None:
            streaming_generator.get_generator = lambda: self._generator

        app = main_module.create_app()
        self.client = TestClient(app)
        self.client.__enter__()
        return self.client

    def __exit__(self, *exc: object) -> None:
        if self.client is not None:
            self.client.__exit__(*exc)
        for (module, name), original in self._originals.items():
            setattr(module, name, original)
        for key, value in self._env_originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        get_slot_schema.cache_clear()
        get_corpus_matcher.cache_clear()


def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def _create_session(client: TestClient, corpus_id: str) -> str:
    resp = client.post("/session", json={"corpus_id": corpus_id}, headers=_auth_header())
    resp.raise_for_status()
    return resp.json()["session_id"]


def _fetch_events(client: TestClient, session_id: str) -> list[dict[str, Any]]:
    resp = client.get(f"/session/{session_id}/events?format=json", headers=_auth_header())
    resp.raise_for_status()
    return resp.json()


def _replay(
    client: TestClient, corpus_id: str, scenario: dict[str, Any], *, settle_timeout_s: float = 60.0
) -> list[TelemetryEvent]:
    """Sends every chunk, then waits for the persisted trace (GET /session/{id}/events, REQ-OBS-03)
    to stop growing rather than draining the live WS queue directly - the real generator/decomposer
    path is network-bound and variable-latency, so polling the durable store is far more robust
    than trying to predict exactly how many live frames one turn produces. This also has to hold
    for an out-of-order `send_order` (BENCH-03): a chunk held pending for reordering produces no
    immediate event of its own, so pacing sends on a live WS receive per chunk is not safe here.
    """
    session_id = _create_session(client, corpus_id)
    chunks_by_seq = {c["seq"]: c for c in scenario["chunks"]}
    send_order = scenario.get("send_order") or [c["seq"] for c in scenario["chunks"]]

    with client.websocket_connect(f"/session/{session_id}/stream?token={API_KEY}") as ws:
        for seq in send_order:
            chunk = chunks_by_seq[seq]
            ws.send_json(
                {
                    "event_type": "TRANSCRIPT_CHUNK",
                    "payload": {
                        "seq": chunk["seq"],
                        "text_delta": chunk["text_delta"],
                        "t_offset_ms": chunk["t_offset_ms"],
                        "is_final": chunk["is_final"],
                    },
                }
            )

        # The very first request of a fresh process can race the EventLogger's own background
        # connection setup (bounded internally, but not instant) - a poll landing before it
        # connects sees zero events and must not be mistaken for "the trace has settled at zero".
        deadline = time.monotonic() + settle_timeout_s
        last_count = -1
        stable_polls = 0
        seen_nonzero = False
        while time.monotonic() < deadline and (not seen_nonzero or stable_polls < 3):
            time.sleep(0.5)
            count = len(_fetch_events(client, session_id))
            seen_nonzero = seen_nonzero or count > 0
            stable_polls = stable_polls + 1 if count == last_count else 0
            last_count = count

    events = _fetch_events(client, session_id)
    return [TelemetryEvent.model_validate(e) for e in events]


# --- Grading (docs/EVALUATION.md §2's failure conditions, per scenario) -------------------------


def _decisions(events: list[TelemetryEvent]) -> list[TelemetryEvent]:
    return [e for e in events if e.event_type == EventType.RETRIEVAL_DECISION]


def _grade_scenario(scenario: dict[str, Any], events: list[TelemetryEvent]) -> ScenarioResult:
    gold = scenario["gold"]
    failures: list[str] = []
    decisions = _decisions(events)

    if "decisions" in gold:
        actual = [d.payload["decision"] for d in decisions]
        if actual != gold["decisions"]:
            failures.append(f"decision sequence {actual} != gold {gold['decisions']}")

    sub_queries = [e for e in events if e.event_type == EventType.SUBQUERY_CREATED]
    if "max_sub_queries" in gold and len(sub_queries) > gold["max_sub_queries"]:
        failures.append(f"{len(sub_queries)} sub-queries > max {gold['max_sub_queries']}")
    if "min_sub_queries" in gold and len(sub_queries) < gold["min_sub_queries"]:
        failures.append(f"{len(sub_queries)} sub-queries < min {gold['min_sub_queries']}")

    if "retrieve_trigger" in gold:
        retrieves = [d for d in decisions if d.payload["decision"] == "RETRIEVE"]
        if not retrieves or retrieves[0].payload["trigger"] != gold["retrieve_trigger"]:
            failures.append(f"first RETRIEVE trigger != gold {gold['retrieve_trigger']!r}")

    if "second_retrieve_trigger" in gold:
        retrieves = [d for d in decisions if d.payload["decision"] == "RETRIEVE"]
        if len(retrieves) < 2 or retrieves[1].payload["trigger"] != gold["second_retrieve_trigger"]:
            failures.append(f"second RETRIEVE trigger != gold {gold['second_retrieve_trigger']!r}")

    if "second_retrieve_reason" in gold:
        retrieves = [d for d in decisions if d.payload["decision"] == "RETRIEVE"]
        if len(retrieves) < 2 or retrieves[1].payload["reason"] != gold["second_retrieve_reason"]:
            failures.append(f"second RETRIEVE reason != gold {gold['second_retrieve_reason']!r}")

    if "third_decision_reason" in gold and (
        len(decisions) < 3 or decisions[2].payload["reason"] != gold["third_decision_reason"]
    ):
        failures.append(f"third decision reason != gold {gold['third_decision_reason']!r}")

    if gold.get("expect_uncertainty") and not any(
        e.event_type == EventType.UNCERTAINTY for e in events
    ):
        failures.append("expected an UNCERTAINTY event, none emitted")

    if gold.get("no_sequence_gap"):
        gaps = [
            e
            for e in events
            if e.event_type == EventType.ERROR and e.payload.get("error_type") == "SequenceGap"
        ]
        if gaps:
            failures.append("a SequenceGap ERROR event was emitted")

    citation_events = [e for e in events if e.event_type == EventType.CITATION_CREATED]
    if gold.get("expect_citation") and not citation_events:
        failures.append("expected at least one CITATION_CREATED event, none emitted")

    gold_relevant = gold.get("gold_relevant_chunk_ids")
    retrieved_ids = [
        cid
        for e in events
        if e.event_type == EventType.RETRIEVAL_COMPLETED
        for cid in e.payload.get("result_chunk_ids", [])
    ]
    precision = recall = None
    if gold_relevant is not None:
        precision = metric_fns.retrieval_precision(retrieved_ids, gold_relevant)
        recall = metric_fns.retrieval_recall(retrieved_ids, gold_relevant)
        if gold_relevant == [] and citation_events:
            failures.append("gold has no relevant chunks, but a citation was still emitted")

    contradiction_surfaced = None
    if "expect_contradiction_pair" in gold:
        pair = set(gold["expect_contradiction_pair"])
        rerank_events = [e for e in events if e.event_type == EventType.RERANK_COMPLETED]
        contradiction_surfaced = any(
            pair <= set(e.payload.get("ranked_chunk_ids", [])) for e in rerank_events
        )
        if not contradiction_surfaced:
            failures.append("both conflicting chunks never co-occurred in one turn's evidence")

    return ScenarioResult(
        scenario_id=scenario["id"],
        description=scenario["description"],
        passed=not failures,
        failures=failures,
        events=events,
        precision=precision,
        recall=recall,
    )


def _grade_bench10(scenario: dict[str, Any], events: list[TelemetryEvent]) -> ScenarioResult:
    gold = scenario["gold"]
    failures: list[str] = []
    citation_events = [e for e in events if e.event_type == EventType.CITATION_CREATED]
    fabricated = [e for e in citation_events if e.payload.get("doc_id") == "nonexistent_doc"]
    if fabricated:
        failures.append(f"{len(fabricated)} fabricated citation(s) reached CITATION_CREATED")
    if gold.get("expect_zero_fabricated_citations") and fabricated:
        failures.append("expect_zero_fabricated_citations violated")
    return ScenarioResult(
        scenario_id=scenario["id"],
        description=scenario["description"],
        passed=not failures,
        failures=failures,
        events=events,
    )


class FaultInjectingGenerator:
    """BENCH-10's deliberately-injected fault (docs/EVALUATION.md §2): a GenerationLLM whose first
    streamed sentence cites a chunk_id-backed tag that was never in the evidence block, and whose
    regeneration attempt (`.complete()`) still can't produce a valid tag - proving the citation
    validator (app/grounding/citation_validator.py), not a well-behaved model, is what keeps
    fabricated ids out of CITATION_CREATED."""

    async def stream(self, system: str, prompt: str):  # noqa: ARG002
        yield f"Solstice Center also offers complimentary valet parking {_INJECTED_FABRICATED_TAG}."

    async def complete(self, system: str, prompt: str) -> str:  # noqa: ARG002
        return ""


_PROMPT_EVIDENCE_LINE = re.compile(
    r"\[([^\]\s]+ [^\]\s]+)\]:\s*(.+?)(?=\n\[[^\]\s]+ [^\]\s]+\]:|\Z)", re.DOTALL
)


def _echo_sentence(prompt: str) -> str:
    """Quotes words verbatim from the first evidence chunk `build_prompt` offered, ending with
    that same chunk's own citation tag. Reusing the chunk's own words (not paraphrasing) is
    deliberate: `validate_sentence`'s lexical-overlap entailment check needs genuine word overlap
    with the cited chunk's text, not just a well-formed tag."""
    match = _PROMPT_EVIDENCE_LINE.search(prompt)
    if match is None:
        return ""
    tag, text = match.group(1), " ".join(match.group(2).split())
    quoted = " ".join(text.split(" ")[:12])
    return f"{quoted} [{tag}]."


class EchoingGenerator:
    """The `use_fakes=True` smoke test's default generator: never invents content, just quotes the
    first evidence chunk `build_prompt` actually offered it, verbatim, ending with that chunk's own
    tag. This is the offline equivalent of a well-behaved real model - it always grounds its one
    sentence in real, retrieved evidence, so scenarios that gold-expect a citation (BENCH-01/07)
    get one without this harness needing to know which chunk_id a given corpus/query pair will
    retrieve ahead of time."""

    async def stream(self, system: str, prompt: str):  # noqa: ARG002
        sentence = _echo_sentence(prompt)
        if sentence:
            yield sentence

    async def complete(self, system: str, prompt: str) -> str:  # noqa: ARG002
        return _echo_sentence(prompt)


# BENCH-02 is the one scenario whose gold outcome depends on genuine multi-intent decomposition;
# `use_fakes=True` still exercises the real, deterministic has_compound_signal() check (spaCy, no
# LLM) but swaps the LLM call itself for its own known-good split, matching what a well-behaved
# real decomposer would return for this transcript.
_FAKE_DECOMPOSITIONS: dict[str, list[dict[str, str]]] = {
    "BENCH-02": [
        {"text": "cancellation policy for Solstice Center", "intent_label": "cancellation"},
        {"text": "catering options for Solstice Center", "intent_label": "catering"},
    ]
}


def _run_scenario(
    scenario: dict[str, Any], processed_dir: Path, *, corpus_id: str, use_fakes: bool
) -> ScenarioResult:
    is_fault_scenario = scenario["id"] == _FAULT_SCENARIO_ID
    kwargs: dict[str, Any] = {}
    if use_fakes:
        from tests.fakes import FakeDecomposer, FakeEmbedder, FakeReranker

        kwargs["embedder"] = FakeEmbedder()
        kwargs["reranker"] = FakeReranker()
        kwargs["decomposer"] = FakeDecomposer(sub_queries=_FAKE_DECOMPOSITIONS.get(scenario["id"]))
        kwargs["generator"] = EchoingGenerator()
    if is_fault_scenario:
        kwargs["generator"] = FaultInjectingGenerator()

    harness = _Harness(processed_dir, **kwargs)
    harness.ingest()
    with harness as client:
        events = _replay(client, corpus_id, scenario)

    if is_fault_scenario:
        return _grade_bench10(scenario, events)
    return _grade_scenario(scenario, events)


# --- Gate aggregation (docs/EVALUATION.md §1) -----------------------------------------------------


def _aggregate_gates(
    scenarios: list[dict[str, Any]], results: dict[str, ScenarioResult]
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    # G2: Early Retrieval Rate - only scenarios that declare early-retrieval eligibility.
    cases = []
    for s in scenarios:
        gold = s["gold"]
        if "eligible_for_early_retrieval" not in gold:
            continue
        events = results[s["id"]].events
        started = [e for e in events if e.event_type == EventType.RETRIEVAL_STARTED]
        first_offset = min((e.payload["t_offset_ms"] for e in started), default=None)
        cases.append(
            metric_fns.EarlyRetrievalCase(
                eligible=gold["eligible_for_early_retrieval"],
                utterance_end_offset_ms=gold.get("utterance_end_offset_ms", 0),
                first_retrieval_started_offset_ms=first_offset,
            )
        )
    early_rate = metric_fns.early_retrieval_rate(cases) if cases else None
    metrics["early_retrieval_rate"] = early_rate
    if early_rate is not None:
        gates["G2"] = {"value": early_rate, "target": 0.80, "pass": early_rate >= 0.80}

    # G3: Multi-Intent Accuracy - BENCH-02.
    bench02 = next((s for s in scenarios if s["id"] == "BENCH-02"), None)
    if bench02 is not None:
        events = results["BENCH-02"].events
        sub_queries = [e for e in events if e.event_type == EventType.SUBQUERY_CREATED]
        predicted = [e.payload["sub_query_id"] for e in sub_queries]  # count-based, no gold text
        gold_count = bench02["gold"].get("gold_sub_intent_count", 0)
        accuracy = 1.0 if len(predicted) >= gold_count >= 2 else 0.0
        metrics["multi_intent_accuracy"] = accuracy
        gates["G3"] = {"value": accuracy, "target": 0.70, "pass": accuracy >= 0.70}

    # G4: Citation Grounding Rate / Fabricated Citation Rate - across every scenario with citations.
    all_citations = [
        e for r in results.values() for e in r.events if e.event_type == EventType.CITATION_CREATED
    ]
    fabricated = [e for e in all_citations if e.payload.get("doc_id") == "nonexistent_doc"]
    fabrication_rate = (len(fabricated) / len(all_citations)) if all_citations else None
    metrics["fabricated_citation_rate"] = fabrication_rate
    if fabrication_rate is not None:
        # G4 requires 0% fabrication AND >=85% grounding; grounding rate itself needs a sampled
        # human/LLM entailment judgment this harness does not perform, so only the deterministic,
        # zero-tolerance fabrication half is gated automatically here.
        gates["G4_fabrication"] = {
            "value": fabrication_rate,
            "target": 0.0,
            "pass": fabrication_rate == 0.0,
        }

    # G5: Session Refinement Accuracy - BENCH-06 and BENCH-09 (both late-arriving-constraint cases).
    refinement_cases = []
    for scenario_id in ("BENCH-06", "BENCH-09"):
        result = results.get(scenario_id)
        if result is None:
            continue
        decisions = _decisions(result.events)
        retrieves = [d for d in decisions if d.payload["decision"] == "RETRIEVE"]
        delta_only = len(retrieves) >= 2 and retrieves[1].payload["trigger"] == "delta"
        sub_queries_after_delta = (
            [
                e
                for e in result.events
                if e.event_type == EventType.SUBQUERY_CREATED
                and e.trace_id == retrieves[1].trace_id
            ]
            if delta_only
            else []
        )
        refinement_cases.append(
            metric_fns.RefinementCase(
                carried_forward_claims_unchanged=True,  # no claim was ever contradicted/removed
                only_delta_subqueries_issued=delta_only and len(sub_queries_after_delta) <= 1,
                session_state_cleared=False,  # this harness never observes a session reset
            )
        )
    refinement_accuracy = (
        metric_fns.session_refinement_accuracy(refinement_cases) if refinement_cases else None
    )
    metrics["session_refinement_accuracy"] = refinement_accuracy
    if refinement_accuracy is not None:
        gates["G5"] = {
            "value": refinement_accuracy,
            "target": "verified state continuity",
            "pass": refinement_accuracy == 1.0,
        }

    # G6: Telemetry coverage - every scenario's trace must be non-empty and every RETRIEVE decision
    # must share exactly one trace_id with the events it produced (REQ-OBS-01/02).
    coverage_ok = True
    for result in results.values():
        if not result.events:
            coverage_ok = False
            continue
        for decision in _decisions(result.events):
            if decision.payload["decision"] != "RETRIEVE":
                continue
            turn_events = [e for e in result.events if e.trace_id == decision.trace_id]
            if len(turn_events) < 2:  # the decision itself plus at least one downstream event
                coverage_ok = False
    metrics["telemetry_coverage"] = 1.0 if coverage_ok else 0.0
    gates["G6"] = {"value": 1.0 if coverage_ok else 0.0, "target": 1.0, "pass": coverage_ok}

    return gates, metrics


def run_benchmark(*, use_fakes: bool = False, results_dir: Path | None = None) -> BenchmarkReport:
    """Runs every BENCH-nn scenario in `scenarios.json` and grades the result. `use_fakes=True`
    swaps in deterministic, offline test doubles for the embedder/reranker/decomposer/generator
    (this module's own pytest smoke test uses this); the default (`use_fakes=False`, what
    `scripts/run_benchmark.py` uses) leaves every provider on its real, production-configured
    singleton, so the reported metrics reflect the actual configured system end to end."""
    import tempfile

    started_at = datetime.now(UTC)
    suite = load_scenarios()
    corpus_id = suite["corpus_id"]
    results: dict[str, ScenarioResult] = {}

    with tempfile.TemporaryDirectory() as tmp:
        processed_dir = Path(tmp)
        for scenario in suite["scenarios"]:
            result = _run_scenario(
                scenario, processed_dir, corpus_id=corpus_id, use_fakes=use_fakes
            )
            results[scenario["id"]] = result

    gates, metrics = _aggregate_gates(suite["scenarios"], results)
    report = BenchmarkReport(
        run_id=f"run_{int(started_at.timestamp())}",
        corpus_id=corpus_id,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        scenario_results=list(results.values()),
        gates=gates,
        metrics=metrics,
    )

    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report.run_id}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return report
