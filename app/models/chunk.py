"""Chunk model (PRD_TRD.md §7.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import new_id


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=new_id)
    doc_id: str
    section: str
    text: str
    token_count: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    bm25_tokens: list[str] = Field(default_factory=list)
