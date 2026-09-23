"""FastAPI app factory and process entrypoint.

App construction, structured logging setup, and the health endpoints required by REQ-DEPLOY-01
(/health liveness, /ready readiness — now also gating on embedder warmup, post-Phase-2 sync).
Business-logic routers (app/api/session.py, stream.py, events.py, evaluate.py) and corpus
ingestion (scripts/ingest_corpus.py) are later phases and are not wired in here yet.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.embeddings import get_embedder

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()


async def _warm_embedder(app: FastAPI) -> None:
    """Load the embedding model into memory once at startup (REQ-DEPLOY-01), so the first real
    query never pays the model-load cost against EMBEDDING_TIMEOUT_MS / the TTFT budget
    (PRD_TRD.md §5.3). Runs as a background task, not awaited by lifespan, so it never delays
    the port bind or /health — /ready gates on its completion instead."""
    try:
        await asyncio.to_thread(get_embedder().warmup)
        app.state.embedder_warm = True
        log.info("embedder_warm")
    except Exception:
        log.exception("embedder_warmup_failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("startup", embedding_model=settings.embedding_model, qdrant_url=settings.qdrant_url)
    app.state.embedder_warm = False
    warmup_task = asyncio.create_task(_warm_embedder(app))
    # Phase 10 (PRD_TRD.md §10) wires scripts/ingest_corpus.py's ingest_corpus() into this startup
    # event as a background task, per REQ-DEPLOY-01 (async ingestion, never blocking the port
    # bind), and sets this flag from its outcome. Until then ingestion runs via the CLI.
    app.state.ingestion_complete = True
    yield
    warmup_task.cancel()
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
        """Readiness — gates on live Qdrant connectivity, ingestion having completed once, and
        the embedding model being warmed (REQ-DEPLOY-01)."""
        settings = get_settings()
        qdrant_ok = _check_qdrant(settings.qdrant_url)
        ingestion_ok = bool(getattr(app.state, "ingestion_complete", False))
        embedder_ok = bool(getattr(app.state, "embedder_warm", False))
        is_ready = qdrant_ok and ingestion_ok and embedder_ok
        body = {
            "status": "ready" if is_ready else "not_ready",
            "qdrant": qdrant_ok,
            "ingestion": ingestion_ok,
            "embedder": embedder_ok,
        }
        return JSONResponse(content=body, status_code=200 if is_ready else 503)

    return app


app = create_app()
