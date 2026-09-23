"""Embedding model (PRD_TRD.md §7.3). Vector dimension fixed at 384 (bge-small-en-v1.5)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

from app.core.embeddings import embedding_dimension


class Embedding(BaseModel):
    model_config = {"protected_namespaces": ()}  # "model_name" is a spec-required field (§7.3)

    chunk_id: str
    vector: list[float]
    model_name: str
    created_at: datetime

    @field_validator("vector")
    @classmethod
    def _check_dimension(cls, v: list[float]) -> list[float]:
        expected = embedding_dimension()
        if len(v) != expected:
            raise ValueError(f"vector must have {expected} dimensions, got {len(v)}")
        return v
