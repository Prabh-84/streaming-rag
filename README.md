# Streaming Live RAG

Full specification: [`PRD_TRD.md`](PRD_TRD.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/API.md`](docs/API.md), [`docs/TELEMETRY.md`](docs/TELEMETRY.md), [`docs/EVALUATION.md`](docs/EVALUATION.md). Those documents are the frozen source of truth; this file only covers running what currently exists.

## Status

**Phases 1–2 complete.** Foundation (config, data models, telemetry envelope, health/readiness, Docker/CI) plus corpus ingestion and retrieval: per-corpus `slots.yaml` validation, chunking with deterministic ids, bge-small embeddings, Qdrant dense index, per-corpus BM25 sparse index, and a hybrid (dense + sparse) retrieval primitive, all scoped by `corpus_id`. Not yet implemented: retrieval controller, decomposition, fusion/reranking, grounding, session refinement, generation, API routes — see `PRD_TRD.md` §10.

No production corpus has been supplied yet; see [`data/corpus/README.md`](data/corpus/README.md) for where it goes.

## Local development (without Docker)

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.lock
./.venv/Scripts/pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
cp .env.example .env
./.venv/Scripts/python -m pytest
./.venv/Scripts/python -m ruff check .
./.venv/Scripts/uvicorn app.main:app --reload
```

The spaCy model is installed as a direct pip wheel rather than via `python -m spacy download` — on some networks the `spacy download` CLI's redirect target (`release-assets.githubusercontent.com`) resets the connection while the same wheel installs cleanly through plain pip. Use whichever succeeds in your environment.

## Docker

`docker-compose.yml` lives under `docker/`, so reference it explicitly:

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

This starts two services: `app` (FastAPI on `:8000`) and `qdrant` (`:6333`). Verify:

```bash
curl http://localhost:8000/health   # {"status":"ok"} — liveness, no dependency checks
curl http://localhost:8000/ready    # {"status":"ready","qdrant":true,"ingestion":true} once Qdrant is reachable
```

## Corpus ingestion

Put a corpus at `data/corpus/<corpus_id>/` (layout: [`data/corpus/README.md`](data/corpus/README.md)), then:

```bash
python scripts/ingest_corpus.py --corpus-id <corpus_id>   # or --all; --force to rebuild
# inside Docker:
docker compose -f docker/docker-compose.yml exec app python scripts/ingest_corpus.py --all
```

Outputs: Qdrant points in `corpus_chunks` (payload carries `corpus_id`), plus `data/processed/bm25__<corpus_id>.pkl`, `chunks__<corpus_id>.jsonl`, and `manifest__<corpus_id>.json`. An unchanged corpus re-ingests as a no-op. Malformed input (missing/invalid `slots.yaml`, non-UTF-8 documents, no supported documents) exits 1 with a structured error and writes nothing.

## Tests and lint

```bash
./.venv/Scripts/python -m pytest -v
./.venv/Scripts/python -m ruff check .
./.venv/Scripts/python -m ruff format --check .
```

Unit tests use a deterministic fake embedder and Qdrant's in-memory mode — no model download, no server. Two opt-in integration suites exercise the real pieces:

```bash
docker compose -f docker/docker-compose.yml up -d qdrant   # tests/integration/test_qdrant_server.py runs when this is up
RUN_MODEL_TESTS=1 ./.venv/Scripts/python -m pytest tests/integration/test_embedding_model.py   # downloads bge-small
```

## Benchmark run, production deployment

Not yet implemented (`scripts/run_benchmark.py` is Phase 9; deployment hardening is Phase 10), per `docs/EVALUATION.md` §7 (Definition of Done).
