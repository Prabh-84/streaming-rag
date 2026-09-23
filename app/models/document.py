"""Document model (PRD_TRD.md §7.1). corpus_id is new in this freeze (REQ-CORPUS-02)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models import new_id


class Document(BaseModel):
    doc_id: str = Field(default_factory=new_id)
    title: str
    source_path: str
    corpus_id: str
    ingested_at: datetime
    content_hash: str
    section_count: int = Field(ge=0)
