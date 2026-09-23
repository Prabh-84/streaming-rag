"""Document model (PRD_TRD.md §7.1). corpus_id is new in this freeze (REQ-CORPUS-02).

doc_id is deterministic and path-derived, not a ULID (PRD_TRD.md §7.1 post-freeze note).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator

from app.core.slots import validate_corpus_id

_DOC_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._/-]")


class Document(BaseModel):
    doc_id: str = Field(min_length=1)
    title: str
    source_path: str
    corpus_id: str
    ingested_at: datetime
    content_hash: str
    section_count: int = Field(ge=0)

    @field_validator("corpus_id")
    @classmethod
    def _check_corpus_id(cls, v: str) -> str:
        return validate_corpus_id(v)

    @staticmethod
    def make_doc_id(relative_path: str | PurePosixPath) -> str:
        """Path relative to the corpus directory, extension dropped, unsafe chars -> '_'."""
        stem = PurePosixPath(relative_path).with_suffix("").as_posix()
        return _DOC_ID_UNSAFE.sub("_", stem)
