"""Per-corpus domain-slot schema loader (REQ-CORPUS-01, resolution #2).

Domain-slot vocabulary (e.g. location, capacity, date) is corpus-supplied data, not application
code, so the entity extractor and multi-intent decomposer stay corpus-agnostic. Each corpus
directory carries `slots.yaml`; this module loads and validates it and caches the result per
corpus_id for the life of the process.

Validation here is structural only. The meaning of a pattern string is defined by the Phase 3
entity extractor (REQ-NLP-01), so this module deliberately does not parse pattern syntax.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import get_settings

SLOTS_FILENAME = "slots.yaml"

_VALID_VALUE_TYPES = {"numeric", "categorical", "date", "text"}
# corpus_id becomes part of filesystem paths (corpus dir, BM25/manifest filenames), so it is
# restricted to a safe charset: this is what blocks path traversal such as "../etc".
_CORPUS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SLOT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ALLOWED_TOP_LEVEL_KEYS = {"slots", "corpus_id", "description"}


class InvalidCorpusIdError(ValueError):
    """corpus_id is not a safe identifier."""


def validate_corpus_id(corpus_id: object) -> str:
    if not isinstance(corpus_id, str) or not _CORPUS_ID_RE.fullmatch(corpus_id):
        raise InvalidCorpusIdError(
            f"invalid corpus_id {corpus_id!r}: must match {_CORPUS_ID_RE.pattern}"
        )
    return corpus_id


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    patterns: list[str]
    value_type: str = "text"

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not _SLOT_NAME_RE.fullmatch(v):
            raise ValueError(f"slot name {v!r} must match {_SLOT_NAME_RE.pattern}")
        return v

    @field_validator("value_type")
    @classmethod
    def _check_value_type(cls, v: str) -> str:
        if v not in _VALID_VALUE_TYPES:
            raise ValueError(f"value_type must be one of {sorted(_VALID_VALUE_TYPES)}, got {v!r}")
        return v

    @field_validator("patterns")
    @classmethod
    def _check_patterns(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("slot must declare at least one pattern")
        if any(not p.strip() for p in v):
            raise ValueError("slot patterns must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError("slot patterns must be unique")
        return v


class SlotSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_id: str
    slots: list[Slot] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_names(self) -> SlotSchema:
        names = self.slot_names()
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate slot names: {duplicates}")
        return self

    def slot_names(self) -> list[str]:
        return [s.name for s in self.slots]


class SlotSchemaError(ValueError):
    """Raised when a corpus's slots.yaml is missing or invalid (ingest_corpus.py fails fast)."""


def slots_path(corpus_id: str, corpus_root: str | Path | None = None) -> Path:
    root = Path(corpus_root) if corpus_root is not None else Path(get_settings().corpus_root)
    return root / validate_corpus_id(corpus_id) / SLOTS_FILENAME


def load_slot_schema(corpus_id: str, corpus_root: str | Path | None = None) -> SlotSchema:
    """Load and validate slots.yaml for one corpus. Raises SlotSchemaError if missing/invalid
    (REQ-CORPUS-01 failure behavior — ingestion must refuse to proceed, not silently skip slots).
    """
    try:
        path = slots_path(corpus_id, corpus_root)
    except InvalidCorpusIdError as exc:
        raise SlotSchemaError(str(exc)) from exc

    if not path.is_file():
        raise SlotSchemaError(f"slots.yaml not found for corpus_id={corpus_id!r} at {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} is not valid UTF-8: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} is not valid YAML: {exc}"
        ) from exc

    if not isinstance(raw, dict) or "slots" not in raw:
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} must be a mapping with a top-level 'slots' key"
        )

    unknown = sorted(set(raw) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise SlotSchemaError(
            f"slots.yaml for corpus_id={corpus_id!r} has unknown top-level keys: {unknown}"
        )

    declared = raw.get("corpus_id")
    if declared is not None and declared != corpus_id:
        raise SlotSchemaError(
            f"slots.yaml declares corpus_id={declared!r} but lives in corpus {corpus_id!r}"
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
