"""EvaluationResult model (PRD_TRD.md §7.11)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.models import new_id


class EvaluationRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class EvaluationResult(BaseModel):
    run_id: str = Field(default_factory=new_id)
    test_set: str
    corpus_id: str
    gates: dict[str, object] = Field(default_factory=dict)
    metrics: dict[str, object] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime | None = None
    status: EvaluationRunStatus = EvaluationRunStatus.QUEUED
