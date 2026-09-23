"""SessionState model (PRD_TRD.md §7.8).

This is the Pydantic validation/transport shape only. The authoritative in-process store (dict,
SQLite-mirrored, per-session asyncio.Lock, TTL sweep) is app/session/session_store.py — Phase 6,
not part of this Phase 1 foundation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.core.embeddings import embedding_dimension


class ClaimStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class Claim(BaseModel):
    claim_id: str
    text: str
    chunk_id: str
    origin_version: int = Field(ge=1)
    status: ClaimStatus = ClaimStatus.ACTIVE


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"
    EXPIRED = "expired"


class SessionState(BaseModel):
    session_id: str
    corpus_id: str
    topic_embedding: list[float] = Field(default_factory=list)
    entities: dict[str, str] = Field(default_factory=dict)
    claims: list[Claim] = Field(default_factory=list)
    answer_versions: list[str] = Field(default_factory=list)
    retrieval_log: list[str] = Field(default_factory=list)
    created_at: datetime
    last_active_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE

    @field_validator("topic_embedding")
    @classmethod
    def _check_dimension(cls, v: list[float]) -> list[float]:
        expected = embedding_dimension()
        if v and len(v) != expected:
            raise ValueError(f"topic_embedding must have {expected} dimensions, got {len(v)}")
        return v
