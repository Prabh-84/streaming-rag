"""POST /session, GET /session/{id} (docs/API.md §2, §5).

Business logic stays in app.session.session_store / app.retrieval — this router only validates
input, delegates, and shapes the response (original TRD §4.2: "Contains no business logic").
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.deps import require_api_key
from app.core.config import get_settings
from app.core.slots import InvalidCorpusIdError
from app.retrieval.sparse_bm25 import SparseIndexRegistry
from app.session.session_store import SessionExpiredError, SessionStore, UnknownSessionError

router = APIRouter(dependencies=[Depends(require_api_key)])


class CreateSessionRequest(BaseModel):
    corpus_id: str | None = None
    client_meta: dict = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    ws_url: str


@router.post("/session", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(payload: CreateSessionRequest, request: Request) -> CreateSessionResponse:
    settings = get_settings()
    corpus_id = payload.corpus_id or settings.default_corpus_id

    sparse: SparseIndexRegistry = request.app.state.sparse_index
    try:
        known = sparse.is_available(corpus_id)
    except InvalidCorpusIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not known:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"unknown corpus_id: {corpus_id!r}"
        )

    store: SessionStore = request.app.state.session_store
    record = await store.create(corpus_id)
    return CreateSessionResponse(
        session_id=record.session_id,
        created_at=record.created_at,
        ws_url=f"/session/{record.session_id}/stream?token={settings.api_key}",
    )


@router.get("/session/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    store: SessionStore = request.app.state.session_store
    try:
        record = await store.get(session_id)
    except (UnknownSessionError, SessionExpiredError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="unknown session"
        ) from exc

    return {
        "session_id": record.session_id,
        "corpus_id": record.corpus_id,
        "status": record.status.value,
        "entities": dict(record.entities),
        "latest_answer_version": len(record.answer_versions) or None,
        "created_at": record.created_at.isoformat(),
        "last_active_at": record.last_active_at.isoformat(),
    }
