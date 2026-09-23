"""SubQuery model (PRD_TRD.md §7.5)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models import new_id


class SubQuery(BaseModel):
    sub_query_id: str = Field(default_factory=new_id)
    session_id: str
    text: str
    intent_label: str
    parent_transcript_offset: int = Field(ge=0)
    created_at: datetime
    merged_from: list[str] = Field(default_factory=list)
