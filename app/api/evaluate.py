"""POST /evaluate, GET /evaluate/{run_id} (docs/API.md §7-8; REQ-EVAL-01).

Triggers the held-out benchmark suite and reports its G1-G6 gate results. This router only
accepts the request, runs it off the event loop, and shapes the response - it holds no benchmark
scenario data, no gold labels, and no BENCH-nn identifiers of its own (REQ-EVAL-01/HC-2: that
content lives exclusively under `benchmarks/`, imported here only as a lazily-resolved runner,
the same "heavy/optional dependency imported inside the function, not at module top-level"
convention this codebase already uses for `anthropic`/`google.genai`/`sentence_transformers`).

A run is CPU/network-bound and can take longer than one request is willing to wait for, so it is
queued (202) and polled (GET), exactly as docs/API.md §7-8 specify - never run synchronously
inside the POST handler.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.api.deps import require_api_key, require_eval_key
from app.models.evaluation_result import EvaluationResult, EvaluationRunStatus

router = APIRouter(dependencies=[Depends(require_api_key), Depends(require_eval_key)])


class TriggerEvaluationRequest(BaseModel):
    test_set: str
    corpus_id: str = "default"


class TriggerEvaluationResponse(BaseModel):
    run_id: str
    status: EvaluationRunStatus


def _run_benchmark_sync() -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs off the event loop (asyncio.to_thread, see below) - benchmarks.harness.run_benchmark
    is itself a blocking call (it drives a real WebSocket replay per scenario). This repository
    ships exactly one held-out suite/corpus pairing, so `test_set`/`corpus_id` are accepted for
    wire-contract shape (docs/API.md §7) and recorded on the EvaluationResult, but do not yet
    select among multiple suites - an honest simplification, not a silent bug."""
    from benchmarks import harness  # lazy: see module docstring

    report = harness.run_benchmark()
    return report.gates, report.metrics


async def _execute(app_state: Any, run_id: str) -> None:
    runs: dict[str, EvaluationResult] = app_state.evaluation_runs
    result = runs[run_id]
    result.status = EvaluationRunStatus.RUNNING
    try:
        gates, metrics = await asyncio.to_thread(_run_benchmark_sync)
        result.gates = gates
        result.metrics = metrics
        result.status = EvaluationRunStatus.COMPLETE
    except Exception as exc:
        result.status = EvaluationRunStatus.FAILED
        result.metrics = {"error": str(exc)}
    finally:
        result.completed_at = datetime.now(UTC)


@router.post("/evaluate", status_code=status.HTTP_202_ACCEPTED)
async def trigger_evaluation(
    payload: TriggerEvaluationRequest, request: Request
) -> TriggerEvaluationResponse:
    runs: dict[str, EvaluationResult] = request.app.state.evaluation_runs
    in_flight = (EvaluationRunStatus.QUEUED, EvaluationRunStatus.RUNNING)
    if any(r.test_set == payload.test_set and r.status in in_flight for r in runs.values()):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a run is already in progress for test_set={payload.test_set!r}",
        )

    result = EvaluationResult(
        test_set=payload.test_set,
        corpus_id=payload.corpus_id,
        started_at=datetime.now(UTC),
    )
    runs[result.run_id] = result
    task = asyncio.create_task(_execute(request.app.state, result.run_id))
    request.app.state.evaluation_tasks.add(task)
    task.add_done_callback(request.app.state.evaluation_tasks.discard)
    return TriggerEvaluationResponse(run_id=result.run_id, status=result.status)


@router.get("/evaluate/{run_id}")
async def get_evaluation(run_id: str, request: Request) -> EvaluationResult:
    runs: dict[str, EvaluationResult] = request.app.state.evaluation_runs
    result = runs.get(run_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown run_id")
    return result
