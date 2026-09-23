"""Citation model (PRD_TRD.md §7.7)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import new_id


class Citation(BaseModel):
    citation_id: str = Field(default_factory=new_id)
    answer_version_id: str
    chunk_id: str
    doc_id: str
    section: str
    claim_text: str
    entailment_score: float
