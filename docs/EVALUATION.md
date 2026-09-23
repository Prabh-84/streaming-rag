# Streaming Live RAG — Evaluation & Benchmark Specification

Canonical source for Gate G1–G6 definitions, the REQ-ID↔test↔gate mapping, benchmark scenario structure, ablation requirements, and submission deliverables. Referenced from `../PRD_TRD.md` §12. Metric formulas live in `TELEMETRY.md` §5 — this file states thresholds and pass/fail logic, not the formulas themselves, to keep each schema in exactly one place.

## 0. Hard rule: no benchmark hardcoding (HC-2, REQ-EVAL-01)

The held-out benchmark replay set (`benchmarks/streaming_suite_v1/`) is private. No prompt, sub-query, or answer text may be embedded in `/app` application code, and no application code may branch on a specific benchmark test-case identifier. `scripts/run_benchmark.py` and `benchmarks/harness.py` supply transcripts and corpus only; all grading happens externally, by reading the `TelemetryEvent` log per the formulas in `TELEMETRY.md` §5. CI enforces this with a grep-based check (`test_no_benchmark_strings_in_app`) scanning `/app` for any literal string from the fixture set.

This document's benchmark scenario descriptions below are structural (what class of input, what class of expected decision) — they contain no gold labels, no gold chunk IDs, and no gold answer text, consistent with this rule.

## 1. Gates G1–G6

| Gate | Criterion | Target threshold | Validation method | Primary REQ-IDs |
|---|---|---|---|---|
| G1 | Reproducibility | Pass/Fail | `docker compose up --build` launches the full stack (app, qdrant) with no manual steps; automated replay suite completes without intervention | REQ-DEPLOY-01, REQ-DEPLOY-02 |
| G2 | Early Retrieval | ≥80% of eligible queries | Retrieval commences before final-transcript completion on held-out streaming prompts, with a low false-trigger rate on no-retrieval cases | REQ-STREAM-02, REQ-CTRL-01/02 |
| G3 | Multi-Intent Identification | ≥70% of compound queries | Accurately identifies and isolates ≥2 distinct sub-intents in compound test utterances | REQ-INTENT-01/02 |
| G4 | Factual Grounding | ≥85% citation support, 0% fabrication | All sampled factual assertions are supported by cited corpus chunks; zero fabricated/hallucinated document IDs | REQ-GROUND-01/02/03, REQ-EVID-03/04 |
| G5 | Session Refinement | Verified state continuity | Late-arriving constraints narrow or update existing responses without clearing session state or re-executing full-corpus search | REQ-SESS-01/02/03 |
| G6 | Telemetry & Observability | 100% trace coverage | Structured logs capture execution timestamps, retrieval triggers, citations, answer version lineage, and token cost for every session | REQ-OBS-01–05 |

## 2. Benchmark scenarios (structural — no gold data)

| # | Case | Scenario class | Decisions under test | Failure condition |
|---|---|---|---|---|
| BENCH-01 | Simple single-intent | One clear, single-topic question | WAIT then RETRIEVE at utterance end, exactly 1 sub-query | >1 sub-query, or 0 citations |
| BENCH-02 | Compound multi-intent | ≥3 distinct sub-questions in one utterance | RETRIEVE (provisional) then decompose into ≥2 sub-queries | <2 sub-queries identified |
| BENCH-03 | Incomplete streaming | Chunks arrive with realistic gaps, mid-word cutoffs | WAIT across unstable chunks; buffer integrity | any dropped/misordered chunk |
| BENCH-04 | Early-retrieval-beneficial | Stable entity appears mid-utterance | PROVISIONAL_RETRIEVE before `is_final` | retrieval only starts at/after final chunk |
| BENCH-05 | Premature-retrieval-risk | Ambiguous opener, no entity for several chunks | WAIT for all ambiguous chunks | RETRIEVE fires before an actionable entity exists |
| BENCH-06 | Late-arriving constraint | Initial answer exists, then a contrastive constraint arrives | REFINING → DECOMPOSING (delta only) | session state cleared or full corpus re-searched |
| BENCH-07 | No-retrieval formatting | Reformat/repeat request after a prior answer exists | NO_RETRIEVAL (`presentation_restructure`) | any retrieval call issued |
| BENCH-08 | Insufficient evidence | Question outside corpus coverage | RETRIEVE then low/empty fused evidence → explicit uncertainty | a citation emitted anyway |
| BENCH-09 | Conflicting evidence | Corpus has two chunks with differing values for one attribute | Both values surfaced with respective citations, discrepancy noted | answer silently picks one value unqualified |
| BENCH-10 | Citation validation (injected fault) | Draft answer references a `chunk_id` not in the evidence set | Validator strips/regenerates; final answer has 0 fabricated IDs | any output citation_id absent from `evidence_by_id` |

## 3. REQ-ID → module → test → gate mapping

This table is the traceability matrix required by the specification freeze — every requirement in `PRD_TRD.md` §4 has a stable REQ-ID, a named test, and (where applicable) a gate.

| REQ-ID | Module | Test name | Gate |
|---|---|---|---|
| REQ-STREAM-01 | `app/controller/retrieval_controller.py` | BENCH-03 | — |
| REQ-STREAM-02 | `app/controller/retrieval_controller.py` | BENCH-04 | G2 |
| REQ-STREAM-03 | `app/api/stream.py` | `test_reconnect_resumes_state` | — |
| REQ-CTRL-01 | `app/controller/retrieval_controller.py` | BENCH-05 | G2 |
| REQ-CTRL-02 | `app/controller/retrieval_controller.py` | BENCH-04 | G2 |
| REQ-CTRL-03 | `app/controller/retrieval_controller.py` | BENCH-07 | G2 |
| REQ-INTENT-01 | `app/decomposition/multi_intent.py` | BENCH-02 | G3 |
| REQ-INTENT-02 | `app/decomposition/multi_intent.py` | BENCH-01 | G3 |
| REQ-CORPUS-01 | `app/core/slots.py` | `test_slots_schema_loads` | — |
| REQ-CORPUS-02 | `app/retrieval/dense.py`, `sparse_bm25.py` | `test_corpus_isolation` | — |
| REQ-NLP-01 | `app/controller/entity_extraction.py`, `app/decomposition/multi_intent.py` | `test_entity_extraction_deterministic`, `test_compound_signal_detection` | — |
| REQ-EVID-01 | `app/retrieval/hybrid.py` | `test_dense_search`, `test_bm25_search`, `test_parallel_wall_time` | — |
| REQ-EVID-02 | `app/retrieval/fusion.py` | `test_rrf_fusion`, `test_dedup_threshold` | — |
| REQ-EVID-03 | `app/reranking/cross_encoder.py` | `test_rerank_order`, `test_global_evidence_cap`, `test_evidence_cap_preserves_min_per_subintent` | G3, G4 |
| REQ-EVID-04 | `app/retrieval/fusion.py` | BENCH-09 | G4 |
| REQ-SESS-01 | `app/session/delta_engine.py` | BENCH-06 | G5 |
| REQ-SESS-02 | `app/session/delta_engine.py` | BENCH-06 | G5 |
| REQ-SESS-03 | `app/session/session_store.py` | `test_session_isolation` | G5 |
| REQ-SUPPRESS-01 | `app/controller/suppression.py` | BENCH-07 | — |
| REQ-SUPPRESS-02 | `app/controller/suppression.py` | `test_first_turn_suppression_guard` | — |
| REQ-GROUND-01 | `app/grounding/citation_validator.py` | BENCH-10 | G4 |
| REQ-GROUND-02 | `app/grounding/citation_validator.py` | BENCH-08 | G4 |
| REQ-GROUND-03 | `app/grounding/citation_validator.py` | BENCH-09 | G4 |
| REQ-EVAL-01 | CI config | `test_no_benchmark_strings_in_app` | — |
| REQ-OBS-01–04 | `app/telemetry/event_logger.py`, `metrics.py` | `test_event_envelope_complete`, `test_trace_id_threading` | G6 |
| REQ-OBS-05 | `app/models/retrieval_event.py` | `test_retrieval_event_trigger_tagging` | G6 |
| REQ-SEC-01–05 | `app/session/session_store.py`, `app/api/deps.py` | `test_session_isolation` | G5 |
| REQ-SEC-06 | `app/api/deps.py`, `app/api/stream.py` | `test_ws_query_token_auth` | — |
| REQ-DEPLOY-01 | `app/main.py`, `docker/Dockerfile` | `test_health_always_200`, `test_ready_gates_on_ingestion` | G1 |
| REQ-DEPLOY-02 | `.github/workflows/ci.yml` | CI job `lint` | G1 |

## 4. Evaluation methodology

1. Corpus and a held-out streaming test set (transcript-chunk sequences with gold sub-intents, gold relevant chunk IDs, gold refinement points) are loaded by `scripts/run_benchmark.py`. Gold data lives in `benchmarks/streaming_suite_v1/` and is never read by `/app`.
2. Each test case is replayed chunk-by-chunk through `POST /session/{id}/stream`, exactly as a real client would, at recorded timestamps.
3. The harness reads back `GET /session/{id}/events` and computes the ten metrics (`TELEMETRY.md` §5) against gold labels.
4. Results are written to `benchmarks/results/<run_id>.json` and compared against the G1–G6 thresholds (§1); `POST /evaluate` triggers this run over HTTP for CI and automated grading.
5. No prompt, sub-query, or answer is ever hardcoded against the held-out set (§0) — the harness only supplies transcripts and corpus; grading is done externally on the output.

## 5. Ablation requirements (Theme 4 Guide Engineering Deliverables Checklist)

The Benchmarking & Evaluation Report deliverable must include quantitative comparisons beyond the automated G1–G6 gates, specifically at least three analyzed edge-case failures and two architectural ablations:

- **Ablation 1 — Hybrid vs. dense-only retrieval.** Run the benchmark suite with BM25 disabled (dense-only) and compare Retrieval Precision/Recall and Citation Grounding Rate against the full hybrid configuration.
- **Ablation 2 — Rule-based vs. model-based controller.** Compare the deterministic stability+entity controller (REQ-CTRL-01/02, REQ-NLP-01) against an LLM-only retrieval-timing decision on Early Retrieval Rate (G2) and false-trigger rate, to justify the deterministic default (HC-5, Architectural Parsimony).

These ablations are reporting deliverables produced in Phase 9 (Evaluation), not automated CI gates — they do not block G1–G6 pass/fail.

## 6. Submission deliverables checklist (Theme 4 Guide §8)

- [ ] Reproducible repository: source code, pinned dependency lockfiles (`requirements.lock`), `.env.example`, one-command run (`docker compose up --build`).
- [ ] System Architecture Brief, ≤6 pages — `docs/ARCHITECTURE.md`.
- [ ] Benchmarking & Evaluation Report — quantitative G1–G6 results, ≥3 analyzed edge-case failures, the 2 ablations in §5.
- [ ] System Demonstration Video, ≤5 minutes — early retrieval triggering, multi-intent decomposition, late-detail refinement, presentation-query suppression, citation traceability, runtime telemetry.
- [ ] Telemetry & Observability Schema — `docs/TELEMETRY.md`.

The video and the populated Benchmarking Report are produced in Phase 9/10 once the implementation exists; they are listed here for completeness, not as Phase 1 blockers.

## 7. Definition of Done

- [ ] Every REQ-ID in `PRD_TRD.md` §4 is implemented and covered by an automated test named after it (§3 table).
- [ ] `docker compose up --build` launches the full stack with one command, no manual steps.
- [ ] `scripts/ingest_corpus.py` ingests the supplied corpus deterministically (idempotent re-run produces the same chunk/embedding set), per `--corpus-id`.
- [ ] `scripts/run_benchmark.py` runs all 10 benchmark scenarios and reports G1–G6 pass/fail using the exact formulas in `TELEMETRY.md` §5.
- [ ] Telemetry trace exists and is complete for 100% of benchmark sessions.
- [ ] Zero fabricated citations across the sampled factual assertions.
- [ ] README documents local run, corpus ingestion, benchmark run, and the production deployment path.
- [ ] CI pipeline (`.github/workflows/ci.yml`) runs `ruff check .`, `pytest`, and the benchmark suite on every push to main.
