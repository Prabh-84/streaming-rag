# Streaming Live RAG

Full specification: [`PRD_TRD.md`](PRD_TRD.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/API.md`](docs/API.md), [`docs/TELEMETRY.md`](docs/TELEMETRY.md), [`docs/EVALUATION.md`](docs/EVALUATION.md). Those documents are the frozen source of truth; this file only covers running what currently exists.

## Status

**Phase 1 (Foundation) complete.** Config, data models, telemetry envelope, corpus slot-schema loader, health/readiness endpoints, and the Docker/CI skeleton exist. Business logic (controller, decomposition, retrieval, grounding, session refinement, generation) is not implemented yet — see `PRD_TRD.md` §10 for the phase plan.

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

## Tests and lint

```bash
./.venv/Scripts/python -m pytest -v
./.venv/Scripts/python -m ruff check .
```

## Corpus ingestion, benchmark run, production deployment

Not yet implemented (`scripts/ingest_corpus.py`, `scripts/run_benchmark.py` are Phase 2 and Phase 9 respectively). This section will be filled in as those phases land, per `docs/EVALUATION.md` §7 (Definition of Done).
