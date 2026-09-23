"""TRANSCRIPT_CHUNK wire model (PRD_TRD.md §11.4 event catalog; REQ-STREAM-01).

Inbound-only: the client->server WebSocket frame payload. Not one of the 11 canonical persisted
models in §7 (it is transient input, not stored state), so it lives here rather than being forced
into that numbering.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TranscriptChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int = Field(ge=0)
    text_delta: str
    t_offset_ms: int = Field(ge=0)
    is_final: bool = False
