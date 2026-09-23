"""FastAPI app factory and process entrypoint.

Phase 1 scope only: app construction, structured logging setup, and the two health endpoints
required by REQ-DEPLOY-01 (/health liveness, /ready readiness). Business-logic routers
(app/api/session.py, stream.py, events.py, evaluate.py) and corpus ingestion
(scripts/ingest_corpus.py) are Phase 2+ and are not wired in here yet.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import get_settings

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("startup", embedding_model=settings.embedding_model, qdrant_url=settings.qdrant_url)
    # Phase 2 will replace this with a real corpus-ingestion call (scripts/ingest_corpus.py) run
    # as part of this same startup event, per REQ-DEPLOY-01 (async ingestion, never blocking the
    # port bind). No ingestion pipeline exists yet in this Phase 1 foundation.
    app.state.ingestion_complete = True
    yield
    log.info("shutdown")


def _check_qdrant(qdrant_url: str) -> bool:
    """Live, short-timeout connectivity check used by /ready. Never raises."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=2.0)
        client.get_collections()
        return True
    except Exception:
        return False


def create_app() -> FastAPI:
    app = FastAPI(title="Streaming Live RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness only — never depends on Qdrant or ingestion (REQ-DEPLOY-01)."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness — gates on live Qdrant connectivity AND ingestion having completed once."""
        settings = get_settings()
        qdrant_ok = _check_qdrant(settings.qdrant_url)
        ingestion_ok = bool(getattr(app.state, "ingestion_complete", False))
        is_ready = qdrant_ok and ingestion_ok
        body = {
            "status": "ready" if is_ready else "not_ready",
            "qdrant": qdrant_ok,
            "ingestion": ingestion_ok,
        }
        return JSONResponse(content=body, status_code=200 if is_ready else 503)

    return app


app = create_app()
