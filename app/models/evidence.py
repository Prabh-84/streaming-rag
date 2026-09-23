"""Evidence model (PRD_TRD.md §7.6)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import new_id


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=new_id)
    sub_query_id: str
    chunk_id: str
    fusion_score: float
    rerank_score: float | None = None
    duplicate_of: str | None = None
    contradiction_pair_id: str | None = None
