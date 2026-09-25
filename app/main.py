"""FastAPI app factory and process entrypoint.

App construction, structured logging, health endpoints (REQ-DEPLOY-01), and the composition root
that wires the Phase 2 retrieval primitives + Phase 3 session store into app.state for the
session/stream routers. Phase 10 wires scripts/ingest_corpus.py's ingest_corpus() into the
startup event below, per this module's own long-standing comment and PRD_TRD.md §10's Phase 10
insertion list ("ingestion moves to FastAPI startup event").
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from app.api import evaluate as evaluate_router
from app.api import events as events_router
from app.api import session as session_router
from app.api import stream as stream_router
from app.controller import entity_extraction
from app.core.config import Settings, get_settings
from app.core.embeddings import get_embedder
from app.core.slots import InvalidCorpusIdError, SlotSchemaError
from app.retrieval.dense import DenseIndex, create_qdrant_client
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse_bm25 import SparseIndexRegistry
from app.session.session_store import SessionStore
from app.telemetry.event_logger import EventLogger
from scripts.ingest_corpus import IngestionError, discover_corpora, ingest_corpus

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


async def _warm_nlp(app: FastAPI) -> None:
    """Load the spaCy model once at startup, for the same reason as _warm_embedder — a cold
    spaCy load on the first transcript chunk would blow the <150ms time-to-first-retrieval-decision
    budget (PRD_TRD.md §5.3, REQ-STREAM-01)."""
    try:
        await asyncio.to_thread(entity_extraction.warmup)
        app.state.nlp_warm = True
        log.info("nlp_warm")
    except Exception:
        log.exception("nlp_warmup_failed")


async def _ingest_corpora(
    app: FastAPI, qdrant_client: AsyncQdrantClient, embedder: Any, settings: Settings
) -> None:
    """Startup corpus ingestion (REQ-DEPLOY-01; PRD_TRD.md §10's Phase 10 insertion: "ingestion
    moves to FastAPI startup event"). Discovers every corpus_id under CORPUS_ROOT and ingests each
    one, reusing the process-wide embedder singleton and the lifespan's own Qdrant client (never a
    second, throwaway client). Runs as a background task, never blocking the port bind - same
    convention as _warm_embedder/_warm_nlp.

    app.state.ingestion_complete is set True only if every discovered corpus ingests (or is
    already up to date) cleanly. A corpus directory that was never mounted at all - the "forgot to
    provide a corpus" deployment mistake PRD_TRD.md §11's risk register names explicitly - leaves
    this False forever, which is exactly what makes /ready never turn ready for that failure mode
    (its whole reason for existing) rather than the container silently serving an empty index.
    """
    corpus_ids = discover_corpora(settings.corpus_root)
    if not corpus_ids:
        log.error("no_corpora_found", corpus_root=settings.corpus_root)
        return
    for corpus_id in corpus_ids:
        try:
            await ingest_corpus(
                corpus_id, settings=settings, embedder=embedder, qdrant_client=qdrant_client
            )
        except (IngestionError, SlotSchemaError, InvalidCorpusIdError) as exc:
            log.error("ingestion_failed", corpus_id=corpus_id, error=str(exc))
            return
        except Exception as exc:  # backend failures: Qdrant unreachable, model load, ...
            log.error(
                "ingestion_failed",
                corpus_id=corpus_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return
    app.state.ingestion_complete = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("startup", embedding_model=settings.embedding_model, qdrant_url=settings.qdrant_url)

    app.state.embedder_warm = False
    app.state.nlp_warm = False
    warmup_tasks = [
        asyncio.create_task(_warm_embedder(app)),
        asyncio.create_task(_warm_nlp(app)),
    ]

    # Set True only once _ingest_corpora (below, started once the shared qdrant_client exists)
    # actually completes ingesting every discovered corpus - see that function's docstring.
    app.state.ingestion_complete = False

    # Phase 8 (REQ-OBS-02/04): one process-wide, non-blocking event logger, started before
    # anything that might emit telemetry so no early event is silently dropped.
    event_logger = EventLogger(settings)
    await event_logger.start()
    app.state.event_logger = event_logger

    # Phase 9 (REQ-EVAL-01): in-memory registry for POST/GET /evaluate - runs are process-lifetime
    # only, same as every other piece of app.state here (REQ-SEC-01's "ephemeral store only").
    app.state.evaluation_runs = {}
    app.state.evaluation_tasks = set()

    # Composition root for Phase 3: one shared retriever + session store per process. Construction
    # is cheap (no I/O until first use) and reuses the already-warmed embedder singleton.
    app.state.session_store = SessionStore(settings, sink=event_logger.sink)
    qdrant_client = create_qdrant_client(settings.qdrant_url)
    dense_index = DenseIndex(qdrant_client, get_embedder(), settings)
    sparse_index = SparseIndexRegistry(settings)
    app.state.sparse_index = sparse_index
    app.state.hybrid_retriever = HybridRetriever(dense_index, sparse_index, settings)

    warmup_tasks.append(
        asyncio.create_task(_ingest_corpora(app, qdrant_client, get_embedder(), settings))
    )

    yield

    for task in warmup_tasks:
        task.cancel()
    await qdrant_client.close()
    await event_logger.stop()
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
    app.include_router(session_router.router)
    app.include_router(stream_router.router)
    app.include_router(events_router.router)
    app.include_router(evaluate_router.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness only — never depends on Qdrant or ingestion (REQ-DEPLOY-01)."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness — gates on live Qdrant connectivity, ingestion having completed once, and
        the embedding + spaCy models being warmed (REQ-DEPLOY-01)."""
        settings = get_settings()
        qdrant_ok = _check_qdrant(settings.qdrant_url)
        ingestion_ok = bool(getattr(app.state, "ingestion_complete", False))
        embedder_ok = bool(getattr(app.state, "embedder_warm", False))
        nlp_ok = bool(getattr(app.state, "nlp_warm", False))
        is_ready = qdrant_ok and ingestion_ok and embedder_ok and nlp_ok
        body = {
            "status": "ready" if is_ready else "not_ready",
            "qdrant": qdrant_ok,
            "ingestion": ingestion_ok,
            "embedder": embedder_ok,
            "nlp": nlp_ok,
        }
        return JSONResponse(content=body, status_code=200 if is_ready else 503)

    return app


app = create_app()
