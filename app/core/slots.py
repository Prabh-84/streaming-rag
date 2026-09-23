"""Per-corpus domain-slot schema loader (REQ-CORPUS-01, resolution #2).

Domain-slot vocabulary (e.g. location, capacity, date) is corpus-supplied data, not application
code, so the entity extractor and multi-intent decomposer stay corpus-agnostic. Each corpus
directory carries `slots.yaml`; this module loads and validates it and caches the result per
corpus_id for the life of the process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings

_VALID_VALUE_TYPES = {"numeric", "categorical", "date", "text"}


class Slot(BaseModel):
    name: str
    patterns: list[str] = Field(default_factory=list)
    value_type: str = "text"

    @field_validator("value_type")
    @classmethod
    def _check_value_type(cls, v: str) -> str:
        if v not in _VALID_VALUE_TYPES:
            raise ValueError(f"value_type must be one of {_VALID_VALUE_TYPES}, got {v!r}")
        return v

    @field_validator("patterns")
    @classmethod
    def _check_patterns_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("slot must declare at least one pattern")
        return v


class SlotSchema(BaseModel):
    corpus_id: str
    slots: list[Slot]

    def slot_names(self) -> list[str]:
        return [s.name for s in self.slots]


class SlotSchemaError(ValueError):
    """Raised when a corpus's slots.yaml is missing or invalid (ingest_corpus.py fails fast)."""


def slots_path(corpus_id: str) -> Path:
    settings = get_settings()
    return Path(settings.corpus_root) / corpus_id / "slots.yaml"


def load_slot_schema(corpus_id: str) -> SlotSchema:
    """Load and validate slots.yaml for one corpus. Raises SlotSchemaError if missing/invalid
    (REQ-CORPUS-01 failure behavior — ingestion must refuse to proceed, not silently skip slots).
    """
    path = slots_path(corpus_id)
    if not path.is_file():
        raise SlotSchemaError(f"slots.yaml not found for corpus_id={corpus_id!r} at {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} is not valid YAML: {exc}"
        ) from exc

    if not isinstance(raw, dict) or "slots" not in raw:
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} must be a mapping with a top-level 'slots' key"
        )

    try:
        return SlotSchema(corpus_id=corpus_id, slots=raw["slots"])
    except Exception as exc:  # pydantic ValidationError or similar
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} failed validation: {exc}"
        ) from exc


@lru_cache(maxsize=32)
def get_slot_schema(corpus_id: str) -> SlotSchema:
    """Cached accessor — one load per corpus_id per process lifetime. Call
    get_slot_schema.cache_clear() only in tests that need to reload after mutating slots.yaml.
    """
    return load_slot_schema(corpus_id)
