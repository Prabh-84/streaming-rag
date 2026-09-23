"""RetrievalEvent model (PRD_TRD.md §7.4) and the hybrid-retrieval request/result shapes.

trigger enum is extended with MULTI_INTENT in this freeze (REQ-OBS-05) — see app.core.events.
RetrievalRequest/RetrievalResult are the input/output of pseudocode 12.C (hybrid_retrieve); the
Phase 5 fusion layer consumes RetrievalResult.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.core.events import RetrievalTrigger
from app.core.slots import validate_corpus_id
from app.models import new_id
from app.models.chunk import RetrievedChunk


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"


class RetrievalEvent(BaseModel):
    event_id: str = Field(default_factory=new_id)
    session_id: str
    sub_query_id: str
    mode: RetrievalMode
    result_chunk_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    t_offset_ms: int = Field(ge=0)
    trigger: RetrievalTrigger


class RetrievalRequest(BaseModel):
    """One sub-query to retrieve for. corpus_id comes from SessionState.corpus_id (§7.8)."""

    session_id: str
    trace_id: str
    sub_query_id: str
    text: str
    corpus_id: str
    trigger: RetrievalTrigger
    t_offset_ms: int = Field(ge=0)

    @field_validator("corpus_id")
    @classmethod
    def _check_corpus_id(cls, v: str) -> str:
        return validate_corpus_id(v)


class RetrievalResult(BaseModel):
    """Dense and sparse ranked lists for one sub-query, kept separate for RRF (Phase 5).

    A failed or timed-out mode contributes an empty list and records why in `<mode>_error`
    (original TRD §7.1: a timed-out sub-query contributes no evidence, is logged, not retried).
    """

    sub_query_id: str
    corpus_id: str
    dense: list[RetrievedChunk] = Field(default_factory=list)
    sparse: list[RetrievedChunk] = Field(default_factory=list)
    dense_error: str | None = None
    sparse_error: str | None = None
    retrieval_events: list[RetrievalEvent] = Field(default_factory=list)
