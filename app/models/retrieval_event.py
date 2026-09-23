"""RetrievalEvent model (PRD_TRD.md §7.4).

trigger enum is extended with MULTI_INTENT in this freeze (REQ-OBS-05) — see app.core.events.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.events import RetrievalTrigger
from app.models import new_id


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
