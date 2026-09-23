"""AnswerVersion model (PRD_TRD.md §7.9)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models import new_id


class AnswerVersion(BaseModel):
    answer_version_id: str = Field(default_factory=new_id)
    session_id: str
    version_no: int = Field(ge=1)
    text: str
    citations: list[str] = Field(default_factory=list)
    supersedes: int | None = None
    carried_forward_claim_ids: list[str] = Field(default_factory=list)
    new_claim_ids: list[str] = Field(default_factory=list)
    created_at: datetime
