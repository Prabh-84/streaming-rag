# Streaming Live RAG

A single FastAPI service that answers a user's question **while they are still speaking**. Instead of waiting for a finished utterance, it streams transcript chunks into a deterministic controller that decides, chunk by chunk, whether enough has been said to retrieve — then decomposes compound questions, retrieves from a corpus-scoped hybrid index, fuses and reranks the evidence, and streams back a grounded, cited answer that it can refine in place as later chunks add new constraints.

Full specification: [`PRD_TRD.md`](PRD_TRD.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/API.md`](docs/API.md), [`docs/TELEMETRY.md`](docs/TELEMETRY.md), [`docs/EVALUATION.md`](docs/EVALUATION.md). Those documents are the frozen source of truth; this file documents what is actually implemented and how to run it.

## Status

**Phases 1–10 complete.** Foundation, corpus ingestion, streaming retrieval controller, multi-intent decomposition, evidence fusion/reranking, session refinement, grounded generation with citations, telemetry/observability, the evaluation/benchmark harness, and Docker/deployment are all implemented and tested. `docker compose up --build` is a genuine one-command deployment: it builds the current codebase into an image, starts `app` + `qdrant`, ingests every corpus under `data/corpus/` at startup, warms the embedding/spaCy models, and `/ready` only turns healthy once all of that has actually happened — verified end to end in a real container, including real Qdrant retrieval, reranking, real Gemini generation, grounded citations, telemetry, and session resync.

No production corpus has been supplied yet; see [`data/corpus/README.md`](data/corpus/README.md) for where it goes. A synthetic benchmark corpus used only by the evaluation harness lives separately under `benchmarks/streaming_suite_v1/`.

## 1. Overview and purpose

Real conversational assistants that wait for a full utterance before doing anything pay a latency tax the user can feel. Streaming Live RAG's premise is that a live transcript already carries enough signal — a stabilizing embedding, a resolved entity — to start retrieval before the speaker finishes, without either retrieving on every noisy partial chunk or fabricating an answer from incomplete evidence.

The system is one deterministic FastAPI process (no agent orchestration, no multi-service fan-out beyond the vector store) that, per conversational turn:

1. Decides whether a transcript chunk is stable and specific enough to retrieve for.
2. Splits a compound request ("the cancellation policy **and** the catering options") into independent sub-queries.
3. Retrieves from a single, corpus-scoped hybrid (dense + sparse) index, fuses and reranks the results, and flags any conflicting evidence.
4. Synthesizes a streamed, sentence-by-sentence answer in which every factual sentence carries a citation traceable to a real retrieved chunk — with a deterministic validator that regenerates or replaces any sentence that would fabricate one.
5. Refines that answer in place — rather than restarting — when a later chunk adds a genuine new constraint to an already-established topic.
6. Logs every stage transition as a structured `TelemetryEvent`, replayable per session or per pipeline turn.

## 2. Architecture / pipeline

```text
Client: transcript chunks
   │  TRANSCRIPT_CHUNK (WS, query-token auth)
   ▼
Retrieval Controller ──WAIT────────────────────▶ (loop: wait for more chunks)
   │
   ├──NO_RETRIEVAL (presentation-only follow-up)──▶ reuse existing answer, no retrieval
   │
   ├──delta on an established topic──▶ Session Refinement classifier
   │        │                              │
   │        │                new_topic ────┘ (falls through to a fresh RETRIEVE below)
   │        │                refinement ──▶ Multi-Intent Decomposer, scoped to the delta only
   │        │                ambiguous ──▶ NO_RETRIEVAL (never guesses)
   │        ▼
   └──RETRIEVE (provisional / final / forced-after-max-wait)──▶ Multi-Intent Decomposer
                                                                      │  SubQuery × N (parallel)
                                                                      ▼
                                                          Hybrid Retriever (dense + sparse, per corpus_id)
                                                                      ▼
                                                          Evidence Fusion (RRF + dedup + contradiction flag)
                                                                      ▼
                                                          Cross-Encoder Reranker (+ global evidence cap)
                                                                      ▼
                                                          Grounding Validator (citation-tag check + entailment)
                                                                      ▼
                                                          Streaming Generator (sentence-by-sentence, cited)
                                                                      ▼
                                                          Streamed answer + citations (ANSWER_DELTA, ANSWER_VERSION_CREATED)

Every stage ──▶ Telemetry Event Log (async queue, batched SQLite writer, GET /session/{id}/events)
Hybrid Retriever ◀──▶ Qdrant (corpus_id-filtered vector store)
Hybrid Retriever ◀──▶ BM25 sparse index (one pickle per corpus_id)
```

Session Refinement is a controller-level classification (`app/session/delta_engine.py`), not a separate pipeline stage after generation: once a session has an established topic, a later chunk that changes an entity is classified as `refinement` (same topic, new constraint — only the delta is retrieved for), `new_topic` (falls back to a full fresh turn), or `ambiguous` (the controller never guesses; it asks rather than retrieving on a coin flip).

## 3. Capabilities implemented (Phases 1–10)

| Phase | Capability |
|---|---|
| 1 | Pydantic data models, `TelemetryEvent` envelope, centralized `Settings`, `/health` + `/ready`, CI/lint scaffold |
| 2 | Per-corpus `slots.yaml` schema validation, deterministic chunking, bge-small embeddings, Qdrant dense index, per-corpus BM25 sparse index, `scripts/ingest_corpus.py` |
| 3 | Streaming Retrieval Controller: chunk buffering with out-of-order reordering, embedding-stability + entity-delta WAIT/RETRIEVE/NO_RETRIEVAL decisions, presentation-only-query suppression |
| 4 | Multi-Intent Decomposer: deterministic (spaCy dependency-parse) compound-signal detection gating a structured LLM call (Anthropic or Gemini), over-fragmentation merge, LLM-failure fallback |
| 5 | Evidence Fusion (Reciprocal Rank Fusion + dedup + contradiction flagging) and Cross-Encoder Reranking with a global evidence cap |
| 6 | Session Refinement: same-topic delta classification, delta-only sub-query retrieval, no full re-decomposition on a late-arriving constraint |
| 7 | Grounding Validator + Streaming Generator: sentence-by-sentence generation, citation-tag validation against the real evidence set, one regeneration attempt, uncertainty fallback — structurally prevents fabricated citations |
| 8 | Telemetry/Observability: async, non-blocking `EventLogger` (SQLite-backed), `GET /session/{id}/events` (SSE + JSON), `SESSION_RESYNC` on reconnect |
| 9 | Evaluation/Benchmark harness: a 10-scenario held-out benchmark suite, the ten `TELEMETRY.md` metric formulas, `POST`/`GET /evaluate`, `scripts/run_benchmark.py` |
| 10 | Docker/Deployment: real startup corpus ingestion (background task, gates `/ready`), persistent Qdrant + SQLite + BM25 storage via named volumes, `benchmarks/` shipped in the image, CI benchmark job |

## 4. Technology stack

| Layer | Choice |
|---|---|
| Backend framework | FastAPI (REST + WebSocket), Pydantic v2 |
| Language / runtime | Python 3.11+ |
| LLM (decomposition + generation) | Anthropic Claude or Google Gemini, selected by `LLM_PROVIDER` |
| Embedding model | `BAAI/bge-small-en-v1.5` (sentence-transformers, local CPU) |
| NLP (entity/compound-signal extraction) | spaCy (`en_core_web_sm`) — deterministic `PhraseMatcher` + dependency parse, no opaque NER |
| Vector database | Qdrant |
| Sparse retrieval | BM25 (`rank_bm25`), one persisted index per `corpus_id` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Telemetry store | SQLite, written via an async queue + batching background task |
| Logging | `structlog` (JSON) |
| Lint/format | `ruff` |
| Testing | `pytest` + `pytest-asyncio` |
| Containerization | Docker + `docker compose` (`app`, `qdrant`) |

## 5. Repository structure

```text
app/
  api/            session.py (POST/GET /session), stream.py (WS /session/{id}/stream),
                  events.py (GET /session/{id}/events), evaluate.py (POST/GET /evaluate), deps.py (auth)
  core/           config.py (Settings), events.py (TelemetryEvent envelope), embeddings.py, slots.py
  controller/     retrieval_controller.py, entity_extraction.py, suppression.py
  decomposition/  multi_intent.py (compound-signal detection + Anthropic/Gemini decomposer)
  retrieval/      dense.py (Qdrant), sparse_bm25.py, hybrid.py, fusion.py
  reranking/      cross_encoder.py
  session/        session_store.py, delta_engine.py (refinement classification)
  generation/     streaming_generator.py, prompt_builder.py
  grounding/      citation_validator.py
  telemetry/      event_logger.py (async SQLite writer), metrics.py (benchmark formulas)
  models/         Pydantic models for every entity in PRD_TRD.md §7
  main.py         app factory, composition root, /health, /ready

scripts/
  ingest_corpus.py    corpus ingestion CLI
  run_benchmark.py    Phase 9 benchmark CLI

benchmarks/
  harness.py                    replay + grading engine (never imported by /app at module scope)
  streaming_suite_v1/           held-out synthetic corpus + 10 gold-labeled BENCH scenarios
  results/                      benchmark run output (gitignored)

data/
  corpus/         production corpora, one directory per corpus_id (empty until you add one)
  processed/      generated BM25 indexes + manifests (gitignored)

docs/             ARCHITECTURE.md, API.md, TELEMETRY.md, EVALUATION.md — canonical specs
docker/           Dockerfile, docker-compose.yml
tests/            unit/ + integration/, mirroring the app/ layout
```

## 6. Local setup and installation

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.lock
./.venv/Scripts/pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
cp .env.example .env
./.venv/Scripts/python -m pytest
./.venv/Scripts/python -m ruff check .
```

The spaCy model is installed as a direct pip wheel rather than via `python -m spacy download` — on some networks the `spacy download` CLI's redirect target resets the connection while the same wheel installs cleanly through plain pip. Use whichever succeeds in your environment.

`.env.example`'s defaults (`QDRANT_URL=http://qdrant:6333`, `SQLITE_PATH=/data/app.db`) target the Docker Compose network. For local (non-Docker) runs, either start Qdrant separately and point `QDRANT_URL` at it (e.g. `http://localhost:6333`), or run `docker compose -f docker/docker-compose.yml up -d qdrant` and leave the rest local; also override `SQLITE_PATH` to a local path (e.g. `./data/app.db`) since `/data/app.db` is an absolute container path.

## 7. Environment configuration

All settings are centralized in `app/core/config.py` (`Settings`, loaded from `.env`); see `.env.example` for the full list with defaults. The ones most relevant to running the pipeline end to end:

- **`LLM_PROVIDER`** — `anthropic` or `gemini`. Selects which client `app/decomposition/multi_intent.py` and `app/generation/streaming_generator.py` construct for multi-intent decomposition and grounded answer generation.
- **`ANTHROPIC_API_KEY`** — required when `LLM_PROVIDER=anthropic`.
- **`GEMINI_API_KEY`** — required when `LLM_PROVIDER=gemini`.
- **`GEMINI_MODEL`** / **`LLM_MODEL`** — model names for the Gemini and Anthropic paths respectively.
- **`API_KEY`** — bearer token required on every REST/WS endpoint.
- **`EVAL_KEY`** — additional header (`X-Eval-Key`) required by `POST`/`GET /evaluate`, on top of `API_KEY`.

**Real API keys belong only in your local, untracked `.env` file and must never be committed.** `.env` is already gitignored; `.env.example` intentionally ships with empty/placeholder values. Only the LLM call itself needs a real key — every other pipeline stage (embedding, entity extraction, retrieval, reranking) runs on local, non-API models.

## 8. Corpus ingestion and corpus isolation

Put a corpus at `data/corpus/<corpus_id>/` (layout and `slots.yaml` schema: [`data/corpus/README.md`](data/corpus/README.md)) **before building the image** — `docker/Dockerfile` bakes `data/corpus` in at build time, and `app/main.py`'s startup lifespan then discovers and ingests every corpus directory under `CORPUS_ROOT` automatically, as a background task that never blocks the port bind. `/ready`'s `ingestion` field only turns `true` once that has genuinely completed for every discovered corpus; if none is mounted at all, it stays `false` forever (deliberately — see §9) rather than the container silently serving an empty index.

To (re-)ingest without rebuilding the image (e.g. after editing a corpus in a running container), or for local (non-Docker) development:

```bash
python scripts/ingest_corpus.py --corpus-id <corpus_id>   # or --all; --force to rebuild
# inside Docker:
docker compose -f docker/docker-compose.yml exec app python scripts/ingest_corpus.py --all
```

Outputs: Qdrant points in the `corpus_chunks` collection (payload carries `corpus_id`), plus `PROCESSED_DIR/bm25__<corpus_id>.pkl`, `chunks__<corpus_id>.jsonl`, and `manifest__<corpus_id>.json` (persisted at `/data/processed` in Docker, via the same `app_data` volume `SQLITE_PATH` uses — survives container recreation). An unchanged corpus re-ingests as a no-op. Malformed input (missing/invalid `slots.yaml`, non-UTF-8 documents, no supported documents) fails that corpus's ingestion and is logged; it does not crash the process.

**Corpus isolation:** `corpus_id` is fixed at session creation (`POST /session`) and is immutable for that session's lifetime. Every retrieval call — dense (Qdrant payload filter) and sparse (a separate BM25 pickle per `corpus_id`) — scopes exclusively to the session's own `corpus_id`; no query path can return another corpus's chunks. This is exercised directly by `tests/integration/test_corpus_isolation.py` and `tests/unit/test_session_store.py`.

The synthetic fixtures under `tests/fixtures/corpora/` and the held-out benchmark corpus under `benchmarks/streaming_suite_v1/corpus/` are separate, self-contained test/evaluation corpora — never copy them into `data/corpus/`.

## 9. Running the application

**Local:**
```bash
./.venv/Scripts/uvicorn app.main:app --reload
```

**Docker** (`docker-compose.yml` lives under `docker/`, so reference it explicitly):
```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

This starts two services: `app` (FastAPI on `:8000`) and `qdrant` (`:6333`). Verify:

```bash
curl http://localhost:8000/health   # {"status":"ok"} — liveness, no dependency checks
curl http://localhost:8000/ready    # {"status":"ready","qdrant":true,"ingestion":true,"embedder":true,"nlp":true} once warm
```

`/ready` gates on live Qdrant connectivity, ingestion having genuinely completed for every discovered corpus, and both the embedding model and the spaCy model having finished warming up. If `data/corpus/` has no corpus directories at all, `ingestion` stays `false` forever — a deliberate design choice (PRD_TRD.md §11 risk register) so a "forgot to mount a corpus" deployment mistake surfaces as `/ready` never turning healthy, not as a container that starts fine and silently serves an empty index. In local testing, a fresh deployment (real model downloads, one small corpus) reached `ready` in about 40 seconds.

## 10. Main REST/WebSocket usage

Full contract: [`docs/API.md`](docs/API.md). All REST endpoints and the WS handshake require `API_KEY`.

**Create a session:**
```bash
curl -X POST http://localhost:8000/session \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"corpus_id": "default"}'
# {"session_id": "...", "created_at": "...", "ws_url": "/session/.../stream?token=..."}
```

**Stream a transcript** — connect to `ws://localhost:8000/session/{id}/stream?token=$API_KEY` and send `TRANSCRIPT_CHUNK` frames as the user speaks:
```json
{"event_type": "TRANSCRIPT_CHUNK", "payload": {"seq": 0, "text_delta": "I need the cancellation policy", "t_offset_ms": 0, "is_final": false}}
```
The server streams back every `TelemetryEvent` the turn produces on the same socket (`RETRIEVAL_DECISION`, `SUBQUERY_CREATED`, `RETRIEVAL_STARTED/COMPLETED`, `RERANK_COMPLETED`, `CITATION_CREATED`, `ANSWER_DELTA`, `ANSWER_VERSION_CREATED`, `UNCERTAINTY`, `ERROR`, ...). Reconnecting to the same `session_id` emits a `SESSION_RESYNC` frame first, then resumes.

**Read session state:** `GET /session/{id}` returns the current snapshot (`corpus_id`, `status`, `entities`, `latest_answer_version`).

## 11. Telemetry / events API

`GET /session/{id}/events` (REQ-OBS-03) returns the full, ordered `TelemetryEvent` log for a session, persisted independently of any live WebSocket connection (via the async `EventLogger`, `app/telemetry/event_logger.py`). Query params: `trace_id` (one pipeline turn), `event_type`, `since`. Transport: Server-Sent Events by default, or `?format=json` for a single JSON array (what the benchmark harness uses).

```bash
curl "http://localhost:8000/session/{id}/events?format=json" -H "Authorization: Bearer $API_KEY"
```

Full event-type catalog, payload fields, and the ten benchmark metric formulas: [`docs/TELEMETRY.md`](docs/TELEMETRY.md).

## 12. Evaluation / benchmark usage

Run the held-out benchmark suite (`benchmarks/streaming_suite_v1/`, 10 gold-labeled `BENCH-01`..`BENCH-10` scenarios covering single/compound intent, early/late retrieval, session refinement, presentation-only suppression, insufficient/conflicting evidence, and an injected citation-fabrication fault) directly:

```bash
./.venv/Scripts/python scripts/run_benchmark.py               # real configured providers (LLM_PROVIDER, embedder, reranker)
./.venv/Scripts/python scripts/run_benchmark.py --use-fakes   # deterministic, offline (no network, no API key needed)
```

Each run prints a per-scenario pass/fail, the G2–G6 gate results, and the computed metrics, and writes the full report to `benchmarks/results/<run_id>.json` (gitignored — run output, not suite data).

The same suite can also be triggered over HTTP: `POST /evaluate` (requires `API_KEY` **and** `X-Eval-Key: $EVAL_KEY`) queues a run and returns `{"run_id", "status": "queued"}` (`202`); poll `GET /evaluate/{run_id}` for `gates`/`metrics` once `status` is `complete`. Full contract: [`docs/API.md`](docs/API.md) §7–8.

`docker/Dockerfile` copies `app/`, `scripts/`, and `benchmarks/` into the image, so both `POST /evaluate` and `scripts/run_benchmark.py` (via `docker compose exec app`) work inside the built container, not just from a local checkout.

## 13. Phase 9 measured results

The results below are from the offline, deterministic run (`scripts/run_benchmark.py --use-fakes`), which exercises the real controller, decomposition-gating, retrieval, fusion, contradiction-flagging, citation-validation, and telemetry logic end to end through the actual WebSocket/HTTP API — the only faked components are the two external LLM calls (multi-intent decomposition wording and answer generation wording), which is what makes the run deterministic and reproducible without network access.

- **10/10 benchmark scenarios passed** (`BENCH-01` through `BENCH-10`)
- **G2 (Early Retrieval Rate) = 1.0** (target ≥ 0.80) — pass
- **G3 (Multi-Intent Accuracy) = 1.0** (target ≥ 0.70) — pass
- **G4 fabrication rate = 0.0** (target = 0.0) — pass
- **G5 (Session Refinement Accuracy) = 1.0** — pass
- **G6 (Telemetry Coverage) = 1.0** (target = 1.0) — pass
- **393 tests passed, 3 skipped** (full `pytest` suite, unit + integration)

## 14. Real-provider (Gemini) benchmark attempt

A genuine end-to-end **benchmark suite** run was also attempted against the real configured Gemini provider (no `--use-fakes`) — ~10 scenarios' worth of decomposition/generation calls in quick succession. It hit the Gemini API's **free-tier rate limit (HTTP 429 `RESOURCE_EXHAUSTED`, 5 requests/minute)** within the first scenario. This is an external account/quota limitation, not a defect in this codebase — the deterministic fallback-on-LLM-failure path (built in Phase 4) degraded gracefully exactly as designed rather than crashing. **The fake-provider results in §13 are not a substitute for a full real-Gemini *benchmark suite* run**; they measure the deterministic pipeline logic, not real LLM-driven decomposition/generation quality at that request volume.

A **single real request** is a different story: a Phase 10 containerized verification (fresh `docker compose up --build`, real Qdrant, real embedder/reranker, real `GEMINI_API_KEY`) sent one transcript through the live WebSocket API and got a genuine, correctly-grounded Gemini-generated answer with a valid citation back, end to end — retrieval, reranking, generation, citation, telemetry, and session resync all confirmed working against the real provider. The free-tier limit is specifically a *request-rate* ceiling (5/minute), not a "Gemini integration doesn't work" finding. A sustained real-Gemini benchmark *suite* run, or any production traffic beyond a handful of requests per minute, needs a Gemini key with sufficient quota (or `LLM_PROVIDER=anthropic` with a funded Anthropic key).

## 15. Tests and lint

```bash
./.venv/Scripts/python -m pytest -v
./.venv/Scripts/python -m ruff check .
./.venv/Scripts/python -m ruff format --check .
```

Most tests use a deterministic fake embedder and Qdrant's in-memory mode — no model download, no server. A few opt-in integration suites exercise the real pieces:

```bash
docker compose -f docker/docker-compose.yml up -d qdrant   # tests/integration/test_qdrant_server.py runs when this is up
RUN_MODEL_TESTS=1 ./.venv/Scripts/python -m pytest tests/integration/test_embedding_model.py tests/integration/test_reranking_real_model.py   # downloads bge-small / the cross-encoder
```
