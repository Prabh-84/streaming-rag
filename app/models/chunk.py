"""Chunk model (PRD_TRD.md §7.2) and RetrievedChunk, the per-hit view returned by retrieval.

chunk_id is a deterministic UUIDv5, not a ULID (PRD_TRD.md §7.2 post-freeze note): it is the
Qdrant point id, and an unchanged corpus must re-ingest to the identical chunk set.
"""

from __future__ import annotations

import hashlib
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "streaming-live-rag/chunk")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    text: str
    token_count: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    bm25_tokens: list[str] = Field(default_factory=list)

    @field_validator("chunk_id")
    @classmethod
    def _check_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(f"chunk_id must be a UUID (Qdrant point id), got {v!r}") from exc
        return v

    @staticmethod
    def make_chunk_id(
        corpus_id: str, doc_id: str, section: str, chunk_index: int, text: str
    ) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        name = "\x1f".join([corpus_id, doc_id, section, str(chunk_index), digest])
        return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, name))


class RetrievedChunk(BaseModel):
    """One ranked hit from dense or sparse retrieval. Carries everything Synthesis needs to cite
    it (original TRD §8.4: chunk_id, doc_id, section, text, score) plus the corpus it came from."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    corpus_id: str
    section: str
    text: str
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=0)
    score: float
    rank: int = Field(ge=1)
