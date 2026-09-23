# Streaming Live RAG — PRD & TRD (Frozen Specification)

Version: 1.0-frozen · 2026-09-23 · Supersedes `Streaming Live RAG — PRD & TRD.md` as the canonical spec.
Source inputs: `Streaming Live RAG — PRD & TRD.md` (original PRD/TRD) + `Theme 4 Guide_RAG (2).pdf` (assignment brief, Samsung Electronics DX/CTO Research).

This document is the single authoritative requirements source. Detailed API, telemetry, and evaluation schemas live in `docs/API.md`, `docs/TELEMETRY.md`, `docs/EVALUATION.md` respectively — this file references them rather than duplicating them, so there is exactly one place each schema can drift. `docs/ARCHITECTURE.md` is a condensed, standalone brief derived from this document for submission purposes.

## 0. Change log (specification freeze)

This freeze incorporates 10 approved resolutions to ambiguities found during architecture review, plus one contradiction found against the Theme 4 Guide's example telemetry record.

| # | Resolution | REQ-ID(s) added/amended |
|---|---|---|
| 1 | spaCy + `en_core_web_sm` for deterministic entity/compound-signal extraction | REQ-NLP-01 |
| 2 | Per-corpus `slots.yaml` domain-slot schema | REQ-CORPUS-01 |
| 3 | REST keeps `Authorization` header; WebSocket uses query-string token | REQ-SEC-06 |
| 4 | `SESSION_RESYNC` event on WebSocket reconnect | REQ-STREAM-03 |
| 5 | First-turn suppression guard (no prior answer → never suppress) | REQ-SUPPRESS-02 |
| 6 | Global evidence cap `MAX_TOTAL_EVIDENCE=15`, minimum 1 chunk/sub-intent | REQ-EVID-03 |
| 7 | Contradiction handling: always surface both sides, drop "prefer authoritative" branch | REQ-GROUND-03 (amended) |
| 8 | `corpus_id` filtering in Qdrant payload + per-corpus BM25 index file | REQ-CORPUS-02 |
| 9 | Ingestion moved to FastAPI startup event (async); `/health` vs `/ready` split clarified | REQ-DEPLOY-01 |
| 10 | ruff adopted for lint/format, wired into CI | REQ-DEPLOY-02 |
| 11 (contradiction fix) | `RetrievalEvent.trigger` enum gains `multi_intent` to tag sub-query-level retrievals produced by decomposition, matching the Theme 4 Guide's example structured output record (`"trigger": "multi_intent"`), which the original enum (`provisional\|final\|delta\|forced_after_max_wait`) had no value for | REQ-OBS-05 |

No resolution weakens corpus isolation, session isolation, or any G1–G6 gate. Full rationale for each item is in the architecture-review record (prior conversation turns); this document states only the resulting requirement.

**Post-freeze correction (Phase 1 post-check):** `SessionState` (§7.8) was missing `corpus_id`, discovered when implementing the Phase 1 Pydantic model. REQ-CORPUS-02 requires every retrieval call within a session to scope by `corpus_id`, but retrieval happens on later turns, separate from the `POST /session` call that supplies it — `SessionState` is the only object carried across a session's turns, so `corpus_id` must live there or REQ-CORPUS-02 cannot be implemented. Added `corpus_id` to §7.8 and clarified REQ-CORPUS-02's Input line accordingly; no other section changed, no gate affected, no architecture redesign.

**Post-freeze correction (Phase 2):** §7's "IDs are ULIDs unless noted" made `Document.doc_id` and `Chunk.chunk_id` impossible to implement: Qdrant rejects ULID strings as point ids (verified: `ValueError: Point id ... is not a valid UUID`), yet §7.13 requires point id = `chunk_id`; and time-random ULIDs contradict the Definition of Done's "idempotent re-run produces the same chunk/embedding set". §7.1 and §7.2 now note deterministic ids (path-derived `doc_id`, UUIDv5 `chunk_id`). All other ids remain ULIDs; no other section changed, no gate affected.

**Post-freeze correction (Phase 2 sync, item 1 — chunk list filename):** §9's repo tree listed the per-corpus artifact as `chunks.jsonl` (unscoped), left over from before `corpus_id` became a first-class scoping dimension (resolution #8) — inconsistent with the `bm25__<corpus_id>.pkl` naming on the very same line. The implementation's `chunks__<corpus_id>.jsonl` (and `manifest__<corpus_id>.json`, also previously unlisted) is correct and is now what §9 documents. Naming only; no behavior, gate, or architecture change.

**Post-freeze correction (Phase 2 sync, item 2 — RETRIEVAL_COMPLETED telemetry fields):** `docs/TELEMETRY.md`'s event table listed `RETRIEVAL_COMPLETED` as `sub_query_id, mode, result_count, latency_ms` only, omitting `result_chunk_ids` and `scores` — but REQ-OBS-02 (§5.1) already required the trace to cover "retrieved chunk IDs and scores," and the Retrieval Precision/Recall formulas (`docs/TELEMETRY.md` §5) are computed from exactly those fields. The table was incomplete, not the implementation (`app/retrieval/hybrid.py` already emitted them). `docs/TELEMETRY.md` §2 updated; no field removed, no new REQ-ID needed.

**Post-freeze correction (Phase 2 sync, item 3 — embedder warmup):** REQ-DEPLOY-01 amended (§5.4) to require the embedding model be warmed at startup, alongside ingestion, for the reason given there. `app/main.py`'s existing startup event now also does this; `/ready`'s response body gains an `embedder` field (`docs/API.md` §9 updated to match).

## 1. Executive summary & problem (Theme 4 Guide §1, original PRD §1.1–1.2)

Streaming Live RAG is a backend engine that answers a user's spoken or typed request while it is still arriving. It consumes timestamped transcript chunks, decides per chunk whether to retrieve, decomposes compound requests into independent sub-queries, retrieves only from a supplied corpus, fuses and reranks evidence, and streams a grounded answer whose every factual claim carries a corpus citation. Late constraints refine the existing answer instead of restarting the pipeline.

Batch RAG waits for a complete utterance before searching (dead air), treats every turn as fresh (under-specified compound queries), and discards verified work on correction. This requires an explicit model of retrieval *timing* (WAIT/RETRIEVE/NO_RETRIEVAL), *decomposition* (compound → independent sub-queries), and *session-scoped memory* (delta refinement, not restart).

## 2. Hard Constraints (Theme 4 Guide §3) and Goals

### 2.1 Hard Constraints (HC) — non-negotiable, source-of-truth is the Theme 4 Guide

| HC-ID | Constraint | Implementing REQ-IDs |
|---|---|---|
| HC-1 | Corpus Isolation — all evidence from the supplied corpus only; no web scraping, no third-party KB, no parametric-memory factual answers | REQ-GROUND-01, REQ-EVID-01/02 |
| HC-2 | No Hardcoding / No Precomputation — held-out benchmark replay is private; no prompts/queries/canned responses embedded in application code | REQ-EVAL-01 |
| HC-3 | Rigorous Factual Grounding — every factual assertion cites a verifiable chunk ID/section; explicit uncertainty when evidence is insufficient | REQ-GROUND-01, REQ-GROUND-02 |
| HC-4 | Session-Bound State — no cross-session profiling/tracking; memory is ephemeral and scoped to the active session | REQ-SESS-03, REQ-SEC-01/02 |
| HC-5 | Architectural Parsimony — multi-agent/complex orchestration is a cost, not a default; every added component must justify its latency/compute overhead | Design principle — evaluated via `docs/ARCHITECTURE.md` design-decision log and the ablation report (`docs/EVALUATION.md` §5), not an automated gate |

### 2.2 Goals

| ID | Goal | Measured by |
|---|---|---|
| GOAL-1 | Begin retrieval before the utterance ends when intent is stable | Gate G2, ≥80% early-retrieval rate |
| GOAL-2 | Decompose compound utterances into independently searchable sub-queries | Gate G3, ≥70% multi-intent accuracy |
| GOAL-3 | Every factual claim traceable to a corpus chunk ID, zero fabricated IDs | Gate G4, ≥85% grounding, 0% fabrication |
| GOAL-4 | Refine answers from late constraints without clearing session state or re-running full-corpus search | Gate G5, verified state continuity |
| GOAL-5 | 100% structured telemetry coverage of decisions, retrievals, citations, versions, cost, latency | Gate G6 |
| GOAL-6 | One-command reproducible deployment | Gate G1 |

### 2.3 Non-goals

- Not a general-purpose conversational agent — answers only from the supplied corpus (HC-1).
- Not a cross-session personalization system — session memory is destroyed at session end (HC-4).
- Not a multi-agent orchestration framework — five deterministic pipeline stages, not autonomous agents (HC-5).
- Not a production ASR/TTS system — input is transcript chunks (real or simulated).
- Not a general web-search-augmented assistant — no external web knowledge in answers (HC-1).
- Not a fine-tuning/model-training project — off-the-shelf models via inference APIs/local weights only.

## 3. Personas and user journeys

Unchanged from the original PRD (see original document for full persona table): Voice support caller, Chat/live-agent user, Compliance/ops reviewer, Benchmark/evaluation harness. Four core journeys — J1 streaming multi-intent, J2 late-arriving constraint, J3 presentation-only follow-up, J4 insufficient evidence — carried forward unchanged; their acceptance criteria are the REQ-IDs in §4.

## 4. Functional Requirements

Format: REQ-ID, description, input, expected behavior, output, failure behavior, acceptance test. Every REQ-ID maps to exactly one module (§9) and, where applicable, one pseudocode block (§10).

### 4.1 Streaming behavior (REQ-STREAM-*)

**REQ-STREAM-01 — Incremental chunk ingestion**
- Input: `TRANSCRIPT_CHUNK` events `{session_id, seq, text_delta, t_offset_ms, is_final}`.
- Expected: appended to session-scoped rolling buffer, passed to Retrieval Controller within 50ms of receipt.
- Output: updated buffer; controller decision event.
- Failure: out-of-order `seq` buffered/reordered up to 2s; beyond that, applied as-is and logged `ERROR: sequence_gap`.
- Test: BENCH-03.

**REQ-STREAM-02 — No fake streaming**
- Expected: `RETRIEVAL_STARTED.t_offset_ms` strictly less than the `is_final=true` chunk's `t_offset_ms` for any case classified early.
- Failure: if `RETRIEVAL_STARTED.t_offset_ms >= utterance_end_offset_ms`, logged `late_retrieval`, fails BENCH-04.
- Test: BENCH-04, Early Retrieval Rate formula (`docs/TELEMETRY.md` §4).

**REQ-STREAM-03 — WebSocket reconnection (NEW, resolution #4)**
- Description: reconnecting to `/session/{id}/stream` must resume state without replaying transcript chunks.
- Input: WS reconnect to an existing, non-expired `session_id`.
- Expected: server emits exactly one `SESSION_RESYNC` event immediately on accept, carrying the latest `AnswerVersion` and current entity/buffer state, before accepting further `TRANSCRIPT_CHUNK` frames.
- Output: `SESSION_RESYNC` event (schema: `docs/TELEMETRY.md` §3).
- Failure: reconnect to an expired/unknown session closes with code `4404` (no resync attempted).
- Test: `test_reconnect_resumes_state`.

### 4.2 Retrieval decision (REQ-CTRL-*)

**REQ-CTRL-01 — WAIT on unstable intent**
- Expected: if inter-chunk cosine similarity < `STABILITY_THRESHOLD` (0.90) or no actionable entity extracted, decision = WAIT.
- Failure: WAIT past `MAX_WAIT_CHUNKS` (6) forces a decision using best-available partial intent.
- Test: BENCH-05.

**REQ-CTRL-02 — Provisional early retrieval**
- Expected: stable, actionable entity set persisting ≥2 consecutive chunks → decision = RETRIEVE, trigger=`provisional`.
- Failure: if provisional retrieval later proves irrelevant (post-hoc similarity to final intent < 0.5), discarded from answer context, logged `wasted_retrieval` (not surfaced as an error).
- Test: BENCH-04.

**REQ-CTRL-03 — No-retrieval classification**
- Expected: decision = NO_RETRIEVAL with reason ∈ {`presentation_only`, `no_actionable_intent`}.
- Failure: Synthesis's secondary check escalates to `RETRIEVAL_DECISION{decision:"RETRIEVE", trigger:"synthesis_escalation"}` rather than fabricating, if a needed fact is missing.
- Test: BENCH-07.

### 4.3 Multi-intent (REQ-INTENT-*)

**REQ-INTENT-01 — Compound request decomposition**
- Expected: LLM structured-output call returns `sub_queries: [{text, intent_label}]`; pairwise cosine < 0.85 or merged.
- Failure: 0 sub-queries for a transcript with detected compound markers → fall back to whole-transcript single sub-query, log `decomposition_fallback`.
- Test: BENCH-02, Gate G3.

**REQ-INTENT-02 — Anti-over-fragmentation**
- Expected: exactly 1 sub-query when no compound marker and no second distinct entity cluster.
- Failure: >1 sub-query with pairwise cosine > 0.85 auto-merged, logged `over_fragmentation_prevented`.
- Test: BENCH-01.

### 4.4 Corpus-supplied domain configuration (REQ-CORPUS-*) — NEW

**REQ-CORPUS-01 — Per-corpus slot schema (resolution #2)**
- Description: domain-slot definitions used by entity extraction and compound-signal detection must be supplied as corpus data, not hardcoded, so the system stays corpus-agnostic (HC-1, HC-5).
- Input: `data/corpus/<corpus_id>/slots.yaml` — slot name, matcher patterns/keyword lists, optional value type (numeric/categorical/date).
- Expected: loaded once at ingestion and cached at app startup into a `SlotSchema` keyed by `corpus_id`; `entity_extraction.py` and `multi_intent.py` read slot definitions only from this schema, never from inline code.
- Output: in-memory `SlotSchema` per `corpus_id`.
- Failure: `scripts/ingest_corpus.py` refuses to ingest a corpus directory missing `slots.yaml` or containing an invalid schema (fails fast, non-zero exit).
- Test: `test_slots_schema_loads`, `test_ingest_rejects_missing_slots`.

**REQ-CORPUS-02 — Corpus-scoped retrieval indexes (resolution #8)**
- Description: multiple named corpora may be pre-ingested; every retrieval read must be scoped by `corpus_id`, using the same mandatory-scoping pattern already required for `session_id` (REQ-SESS-03).
- Input: `corpus_id` supplied on `POST /session` (default `"default"`); persisted on `SessionState.corpus_id` (§7.8) for the session's lifetime so later turns — arriving as separate `TRANSCRIPT_CHUNK` events, potentially many pipeline turns after session creation — can still scope their retrieval calls without the client resending it.
- Expected: Qdrant collection `corpus_chunks` carries an indexed payload field `corpus_id`; every dense query filters on it. BM25 loads a per-corpus pickle: `data/processed/bm25__<corpus_id>.pkl`.
- Output: retrieval results contain chunks from exactly one `corpus_id`.
- Failure: a query naming an uningested `corpus_id` returns `400 unknown_corpus` at session creation (checked once, not per-retrieval).
- Test: `test_corpus_isolation` (asserts corpus B's chunks never appear in a corpus-A session's results — same class of test as `test_session_isolation`).

### 4.5 Deterministic language processing (REQ-NLP-*) — NEW

**REQ-NLP-01 — Rule-based entity & compound-signal extraction (resolution #1)**
- Description: entity/slot extraction and compound-signal detection (§6.1–6.2 of the original TRD) must use a deterministic, local, rule-based method — never an opaque statistical NER model — to preserve auditability (HC-5) and zero external calls on the timing-critical path.
- Input: transcript buffer text, active `SlotSchema` (REQ-CORPUS-01).
- Expected: spaCy (`en_core_web_sm`) provides tokenization, POS tags, and dependency parse only; slot-filling uses spaCy's `Matcher`/`EntityRuler` compiled from `SlotSchema` patterns; compound-signal detection walks the dependency tree for `cc`/`conj` relations spanning two distinct slot-bound noun phrases.
- Output: `entity_state` dict (slot → value), boolean compound-signal flag.
- Failure: an entity found but not bound to a known slot is treated as no entity (still WAIT per REQ-CTRL-01) — never retrieved on noise.
- Test: `test_entity_extraction_deterministic`, `test_compound_signal_detection`.

### 4.6 Evidence retrieval and fusion (REQ-EVID-*) — NEW (elevating original TRD §7 prose to REQ-ID status)

**REQ-EVID-01 — Hybrid dense+sparse retrieval per sub-query**
- Input: one `SubQuery`.
- Expected: dense (Qdrant, bge-small-en-v1.5, top-`K_DENSE`=20) and sparse (BM25, top-`K_SPARSE`=20) run concurrently per sub-query via `asyncio.gather`; all sub-queries across one utterance also run concurrently, bounded by `MAX_CONCURRENT_RETRIEVALS`=8.
- Output: `RetrievalResult{sub_query_id, dense, sparse}`.
- Failure: per-task timeout `RETRIEVAL_TIMEOUT_MS`=800ms — a timed-out sub-query contributes no evidence, is logged, not retried inline.
- Test: `test_dense_search`, `test_bm25_search`, `test_parallel_wall_time` (asserts wall time ≈ max, not sum, of per-sub-query latency).

**REQ-EVID-02 — RRF fusion and deduplication**
- Input: list of `RetrievalResult`.
- Expected: RRF (`rrf_k`=60) per sub-query and globally across sub-queries; duplicates (`chunk_id` match or cosine > `DEDUP_THRESHOLD`=0.95) collapsed, higher-fused-score instance kept, other retained in a `duplicate_of` map for telemetry.
- Output: deduped, per-sub-query-tagged fused candidate pool.
- Failure: none defined (pure function over already-fetched results).
- Test: `test_rrf_fusion`, `test_dedup_threshold`.

**REQ-EVID-03 — Reranking with global evidence cap (resolution #6)**
- Description: `FINAL_K`=6 chunks per sub-query are kept after cross-encoder reranking, but the combined pool across all sub-queries must not exceed `MAX_TOTAL_EVIDENCE`=15, while guaranteeing every sub-intent that has ≥1 chunk above `MIN_RELEVANCE` keeps at least 1 chunk in the final pool (protects Gate G3/G4 from one dominant sub-query starving another's citations).
- Input: deduped candidate pool (REQ-EVID-02), per-sub-query text.
- Expected: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores (sub_query, chunk) pairs; per-sub-query top-`FINAL_K` kept above `MIN_RELEVANCE`=0.35; a post-processing step `cap_global_evidence` merges all kept lists, and if the merged count exceeds `MAX_TOTAL_EVIDENCE`, truncates by global rerank score while enforcing the ≥1-per-sub-intent floor; truncated further to `EVIDENCE_TOKEN_BUDGET`=3000 tokens.
- Output: final evidence set passed to Synthesis, ≤15 chunks, ≤3000 tokens.
- Failure: a sub-query with 0 chunks above `MIN_RELEVANCE` is flagged `low_evidence` and excluded from the floor guarantee (nothing to guarantee).
- Test: `test_rerank_order`, `test_global_evidence_cap`, `test_evidence_cap_preserves_min_per_subintent`.

**REQ-EVID-04 — Contradiction-pair retention**
- Input: deduped candidate pool.
- Expected: chunks addressing the same (entity, attribute) pair with materially different values flagged as a `contradiction_pair` and both retained through truncation (not silently deduped or top-1'd).
- Output: `contradiction_pair_id` set on affected `Evidence` rows.
- Failure: merging conflicting evidence into one unqualified claim fails BENCH-09.
- Test: BENCH-09; feeds REQ-GROUND-03.

### 4.7 Session refinement (REQ-SESS-*)

**REQ-SESS-01 — Delta-only refinement**
- Expected: classifier labels a new segment `refinement` (topic-similarity ≥0.75 and/or contrastive marker) vs `new_topic`; on `refinement`, only the delta entity/constraint builds new sub-queries.
- Failure: ambiguous classification (similarity 0.4–0.75) → one-line disambiguating question, never a guess.
- Test: BENCH-06, Gate G5.

**REQ-SESS-02 — Fact preservation**
- Expected: each prior claim checked for contradiction against the new constraint; non-contradicted claims copied forward verbatim with original `chunk_id` citations.
- Failure: a contradicted claim is marked `superseded`, never silently dropped.
- Test: BENCH-06.

**REQ-SESS-03 — Session isolation**
- Expected: `SessionState` keyed exclusively by `session_id`; store lookups always scope by that key; purged on close/TTL.
- Failure: any cross-session read is a `SECURITY`-class error, fails the build.
- Test: `test_session_isolation`.

### 4.8 Query suppression (REQ-SUPPRESS-*)

**REQ-SUPPRESS-01 — Presentation-only detection**
- Expected: rule-based pre-filter + LLM confirmation when ambiguous; if `retrieval_required=false`, route to Synthesis operating only on `SessionState.last_answer`.
- Failure: if the suppressed request needs a new fact not in `last_answer`, Synthesis's escalation check issues a targeted retrieval for only the new element.
- Test: BENCH-07.

**REQ-SUPPRESS-02 — First-turn suppression guard (NEW, resolution #5)**
- Description: a presentation-only pattern match with no prior answer to reformat is not a valid suppression case.
- Input: transcript segment, `session.answer_versions`.
- Expected: `check_suppression` returns `None` (not suppressed) whenever `session.answer_versions` is empty, regardless of pattern match; the transcript falls through to normal Controller WAIT/RETRIEVE/NO_RETRIEVAL logic.
- Output: no `NO_RETRIEVAL{reason:"presentation_restructure"}` can be emitted on a session's first turn.
- Failure: none (a guard clause; failure would be an undefined "reformat nothing" state, which this eliminates).
- Test: `test_first_turn_suppression_guard`.

### 4.9 Citation / grounding (REQ-GROUND-*)

**REQ-GROUND-01 — Mandatory citation**
- Expected: Grounding Validator segments the draft into claims, checks each claim's cited `chunk_id` ∈ evidence set for that turn AND passes lexical/entailment overlap against that chunk's text.
- Failure: citation to a `chunk_id` not in the evidence set is fabrication — validator strips it, forces regeneration with `citation_required=true`; if still failing, sentence replaced with an uncertainty statement.
- Test: BENCH-10, Gate G4, Fabricated Citation Rate = 0%.

**REQ-GROUND-02 — Explicit uncertainty**
- Expected: fused evidence top rerank score < `MIN_RELEVANCE` (0.35) or empty → answer includes an `uncertainty` field naming the unresolved sub-intent and, where useful, a clarifying question.
- Failure: omitting the uncertainty note fails BENCH-08.
- Test: BENCH-08.

**REQ-GROUND-03 — Contradiction handling (AMENDED, resolution #7)**
- Description: when retrieved evidence conflicts, both positions must always be surfaced — the previous "or prefer the more authoritative/recent source" branch is removed as a MUST, since no field in the data model supported that comparison and the safer branch is also the one BENCH-09 rewards.
- Input: fused evidence with a `contradiction_pair` flag (REQ-EVID-04).
- Expected: both positions are surfaced in the answer text with their respective citations and an explicit discrepancy note; `Document.ingested_at` (existing field, §7.1) may be used only to decide *display order* (which value is mentioned first), never to suppress either side.
- Output: answer text flags the conflict; `citations[]` includes both sources.
- Failure: merging conflicting evidence into one unqualified claim fails BENCH-09.
- Test: BENCH-09.

### 4.10 Evaluation integrity (REQ-EVAL-*) — NEW (elevating Theme 4 Guide HC-2 to a REQ-ID)

**REQ-EVAL-01 — No benchmark hardcoding**
- Description: the held-out benchmark replay set is private; no prompt, sub-query, or answer text may ever be embedded in application code, and no application code may branch on a specific benchmark test-case identifier.
- Input: N/A (a constraint on the codebase, not a runtime input).
- Expected: `scripts/run_benchmark.py` and `benchmarks/harness.py` supply transcripts and corpus only; grading happens externally, from the `TelemetryEvent` log, per `docs/EVALUATION.md`.
- Output: N/A.
- Failure: any literal benchmark string or test-case-ID branch found in `/app` fails code review and CI (a grep-based CI check for `benchmarks/streaming_suite_v1` fixture strings inside `/app` is required — see `docs/EVALUATION.md` §6).
- Test: `test_no_benchmark_strings_in_app` (CI grep check).

## 5. Observability, Security, Performance, Deployment requirements

Full schemas live in `docs/TELEMETRY.md` (event catalog, metrics) and `docs/API.md` (endpoints, auth transport). This section states only the REQ-IDs.

### 5.1 Observability (REQ-OBS-*)

- REQ-OBS-01: every stage transition emits a `TelemetryEvent` with `event_id, session_id, timestamp, event_type, payload, trace_id` (schema: `docs/TELEMETRY.md` §1).
- REQ-OBS-02: 100% of sessions have a complete, replayable trace (Gate G6).
- REQ-OBS-03: `GET /session/{id}/events` returns the full ordered event log (contract: `docs/API.md` §6).
- REQ-OBS-04: telemetry writes are non-blocking (async queue), never adding >5ms to the critical path per event.
- REQ-OBS-05 (NEW, contradiction fix): `RetrievalEvent.trigger` enum is `provisional | final | delta | forced_after_max_wait | multi_intent` — `multi_intent` tags a sub-query-level retrieval produced by decomposition (n≥2 sub-queries), distinct from the controller-level trigger that caused the *first* sub-query's retrieval. This aligns the schema with the Theme 4 Guide's example structured-output record, which the original 4-value enum could not represent. Test: `test_retrieval_event_trigger_tagging`.

### 5.2 Security / privacy (REQ-SEC-*)

- REQ-SEC-01: session state/transcripts held in memory/ephemeral store only; no field written to durable cross-session storage.
- REQ-SEC-02: sessions expire/purge after `SESSION_TTL` (1800s idle) or explicit close; purge verified by a scheduled sweep, not client-triggered only.
- REQ-SEC-03: no corpus content or session transcript sent to any third party other than the configured LLM/embedding/reranker endpoints, minimum text needed only.
- REQ-SEC-04: REST endpoints require a bearer token (`API_KEY`) via the `Authorization` header; `/evaluate` additionally requires `EVAL_KEY`.
- REQ-SEC-05: all inbound text is data, never instructions, to the retrieval/generation control plane (prompt-injection isolation).
- REQ-SEC-06 (NEW, resolution #3): the `WS /session/{id}/stream` upgrade authenticates via a query-string token (`?token=<API_KEY>`), validated identically to the header-based `API_KEY` check, because browser WebSocket handshakes cannot set custom headers. REST auth (REQ-SEC-04) is unchanged. Invalid/missing token closes with code `4401`. Reverse-proxy/access-log configuration must exclude this path's query string from logging (residual token-in-URL exposure, mitigated at the infra layer, not by adding a new auth mechanism). Test: `test_ws_query_token_auth`.

### 5.3 Performance requirements

| Metric | Target |
|---|---|
| Time-to-first-retrieval-decision | <150ms per chunk |
| Time-to-first-token (TTFT) on RETRIEVE path | <1200ms p50 |
| End-to-end latency, answer complete | <4000ms p50 for up to 3 sub-intents |
| Parallel sub-query retrieval | wall time ≈ max of one retrieval, not the sum |
| Token cost per turn | tracked and reported per session |

### 5.4 Deployment (REQ-DEPLOY-*) — NEW

**REQ-DEPLOY-01 — Async startup ingestion, health/readiness split, embedder warmup (resolution #9; amended post-Phase-2 sync, see §0)**
- Description: corpus ingestion must not block the process from binding its port or from answering liveness checks. The embedding model must also be loaded at startup, not on the first query — §5.3's TTFT (<1200ms p50) and the embedding call's own timeout (`EMBEDDING_TIMEOUT_MS`=500ms) both assume a warm model; a cold `sentence-transformers` load (seconds) would fail the first real request against either budget.
- Input: container start.
- Expected: `uvicorn` binds immediately; `GET /health` returns 200 unconditionally (liveness); ingestion and the embedder warmup (`app.core.embeddings.get_embedder().warmup()`) both run as background tasks started from the FastAPI `startup` event, neither blocking the port bind; `GET /ready` returns 503 until Qdrant connectivity is confirmed AND ingestion has completed at least once AND the embedder has finished warming (readiness).
- Output: `/health` always fast; `/ready` gates real traffic, including the first embedding call.
- Failure: ingestion failure or warmup failure at startup logs `ERROR` and keeps `/ready` at 503 indefinitely (container stays up for diagnosis, doesn't crash-loop).
- Test: `test_health_always_200`, `test_ready_gates_on_qdrant_connectivity`, `test_ready_ok_when_qdrant_ingestion_and_embedder_ready`, `test_ready_stays_not_ready_while_embedder_is_not_yet_warm`, `test_ready_stays_not_ready_when_warmup_fails`.

**REQ-DEPLOY-02 — Lint in CI (resolution #10)**
- Description: Definition of Done requires CI to run lint; tool must be named.
- Expected: `ruff check .` runs in CI before `pytest`; `pyproject.toml` carries `[tool.ruff]` config.
- Failure: lint failure fails the CI build (blocks merge to main).
- Test: CI job `lint`.

## 6. Architecture, algorithms, data flow

Full diagrams and the condensed brief: `docs/ARCHITECTURE.md`. Full pseudocode for all 9 core algorithms (Controller, Decomposer, Hybrid Retriever, Fusion, Reranking incl. `cap_global_evidence`, Grounding Validator incl. amended contradiction handling, Session Delta Refinement, Query Suppression incl. first-turn guard, Streaming Answer Generator) is unchanged in mechanism from the original TRD Part 5 except where this document's §4 REQ-IDs above specify a resolution; those deltas are:

- 12.E (Reranking) gains a `cap_global_evidence(per_subquery_kept, max_total=15)` call after per-sub-query reranking, before token-budget truncation (REQ-EVID-03).
- 12.F (Grounding) contradiction step now always emits both sides; the authority-preference branch is deleted (REQ-GROUND-03).
- 12.H (Suppression) gains a leading guard: `if not session.answer_versions: return None` (REQ-SUPPRESS-02).
- Controller/Decomposer's `extract_entities` and `has_compound_signal` are now specified to run on spaCy dependency parse + `Matcher` against the active corpus's `SlotSchema`, not an unspecified extractor (REQ-NLP-01, REQ-CORPUS-01).

## 7. Data models and vector-store schema

Canonical models (Pydantic in `app/models/`, IDs are ULIDs unless noted). Two fields added by this freeze, marked NEW; everything else unchanged from the original TRD Part 6.

### 7.1 Document
`doc_id` (PK) · `title` · `source_path` · `corpus_id` (str, indexed — **NEW**, resolution #8) · `ingested_at` (datetime) · `content_hash` · `section_count`.

`doc_id` is **not** a ULID (post-freeze correction, see §0): it is the document's path relative to its corpus directory, extension removed, characters outside `[A-Za-z0-9._/-]` replaced by `_` — deterministic across re-ingestion and readable in `[doc_id §section]` citation tags (HC-3).

### 7.2 Chunk
`chunk_id` (PK) · `doc_id` (FK) · `section` · `text` · `token_count` · `chunk_index` · `bm25_tokens`.

`chunk_id` is **not** a ULID (post-freeze correction, see §0): it is a UUIDv5 over `(corpus_id, doc_id, section, chunk_index, sha256(text))` — deterministic, so an unchanged corpus re-ingests to the identical chunk set (Definition of Done), and a valid Qdrant point id (§7.13 requires point id = `chunk_id`; Qdrant accepts only unsigned integers or UUIDs).

### 7.3 Embedding
`chunk_id` (PK/FK) · `vector` (float[384]) · `model_name` · `created_at`. Qdrant point payload+vector; `chunk_id` mirrored in SQLite.

### 7.4 RetrievalEvent
`event_id` (PK) · `session_id` (FK) · `sub_query_id` (FK) · `mode` (dense\|sparse) · `result_chunk_ids` · `scores` · `latency_ms` · `t_offset_ms` · `trigger` (enum: `provisional\|final\|delta\|forced_after_max_wait\|multi_intent` — **enum extended**, REQ-OBS-05).

### 7.5 SubQuery
`sub_query_id` (PK) · `session_id` (FK) · `text` · `intent_label` · `parent_transcript_offset` · `created_at` · `merged_from`.

### 7.6 Evidence
`evidence_id` (PK) · `sub_query_id` (FK) · `chunk_id` (FK) · `fusion_score` · `rerank_score` (nullable) · `duplicate_of` (nullable) · `contradiction_pair_id` (nullable).

### 7.7 Citation
`citation_id` (PK) · `answer_version_id` (FK) · `chunk_id` (FK) · `doc_id` (FK) · `section` · `claim_text` · `entailment_score`.

### 7.8 SessionState
`session_id` (PK) · `corpus_id` (str, indexed — **NEW, post-freeze correction, see §0**) · `topic_embedding` (float[384]) · `entities` (JSON) · `claims` (JSON list) · `answer_versions` (list, FK) · `retrieval_log` (JSON, dedup guard) · `created_at` · `last_active_at` · `status` (active\|closed\|expired).

`corpus_id` is set once at session creation from `POST /session`'s `corpus_id` (default `"default"`) and is immutable for the life of the session — every retrieval call issued during that session (REQ-EVID-01) scopes against this value (REQ-CORPUS-02). It is not a new scoping mechanism: it is the same `session_id`-style mandatory key already required by REQ-SESS-03, applied to corpus selection instead of session selection.

### 7.9 AnswerVersion
`answer_version_id` (PK) · `session_id` (FK) · `version_no` · `text` · `citations` (list, FK) · `supersedes` (nullable) · `carried_forward_claim_ids` · `new_claim_ids` · `created_at`.

### 7.10 TelemetryEvent
`event_id` (PK) · `session_id` (FK) · `trace_id` (indexed) · `timestamp` (indexed) · `event_type` (enum, `docs/TELEMETRY.md` §2) · `payload` (JSON).

### 7.11 EvaluationResult
`run_id` (PK) · `test_set` · `corpus_id` · `gates` (JSON) · `metrics` (JSON) · `started_at` · `completed_at` · `status` (queued\|running\|complete\|failed).

### 7.12 SlotSchema (NEW, resolution #2) — not a DB table; loaded from `data/corpus/<corpus_id>/slots.yaml`
`corpus_id` (key) · `slots: [{name, patterns[], value_type}]`. Cached in-process at startup; reloaded only on explicit corpus re-ingestion.

### 7.13 Vector-store schema (Qdrant)

Collection `corpus_chunks`: vector size 384, distance = Cosine. Payload fields (indexed): `chunk_id` (keyword), `doc_id` (keyword), `corpus_id` (keyword — **NEW**), `section` (keyword), `text` (text, payload-filter only). One point per Chunk, point id = `chunk_id`. BM25 stays outside Qdrant, one pickled inverted index per corpus: `data/processed/bm25__<corpus_id>.pkl`.

## 8. Technical stack (final)

| Layer | Choice | Why |
|---|---|---|
| Backend framework | FastAPI | Native async, WebSocket support, Pydantic validation |
| Language | Python 3.11+ | Ecosystem fit, asyncio sufficient for this I/O-bound pipeline |
| LLM | Anthropic Claude (Messages API, streaming) | Native streaming, structured output for decomposition/citation tagging |
| Embedding model | BAAI/bge-small-en-v1.5 (sentence-transformers, local CPU) | No external call on the timing-critical path |
| NLP (entity/syntax) | **spaCy + en_core_web_sm** (NEW) | Local CPU, deterministic Matcher/EntityRuler + dependency parse; no black-box NER |
| Vector database | Qdrant (self-hosted, docker-compose) | In-compose, payload filtering, mature client |
| Sparse retrieval | BM25 (`rank_bm25`), one persisted index per `corpus_id` | Simple, deterministic, easy RRF fusion |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | CPU-feasible, strong relevance lift over RRF alone |
| API protocol | REST (session lifecycle) + WebSocket (streaming) | REST for create/read; WS for full-duplex |
| Streaming protocol | WebSocket for `/stream`; SSE for `/events` replay | WS bidirectional; SSE simpler for one-way replay |
| State store | SQLite (SQLAlchemy), in-process dict hot path | Zero external dependency, swappable for Postgres |
| Logging | structlog, JSON output | Matches TelemetryEvent envelope |
| Metrics | prometheus-client, `/metrics` | Standard scrape format |
| Lint/format | **ruff** (NEW) | One tool, lint+import-sort+format, minimal footprint |
| Testing | pytest + pytest-asyncio | Standard for async FastAPI |
| Containerization | Docker + docker-compose | Two services only: app, qdrant |
| Deployment (prod) | Single container (Fly.io/Render/Fargate) + managed/self-hosted Qdrant + Postgres swap | No Kubernetes/microservices (HC-5) |

Two-service system (`app`, `qdrant`). BM25 index(es) and SQLite live inside the `app` container's volume.

## 9. Repository structure (final)

```text
/app
  /api
    session.py            # POST /session, GET /session/{id}
    stream.py              # WS /session/{id}/stream (query-token auth, SESSION_RESYNC)
    events.py              # GET /session/{id}/events (SSE)
    evaluate.py             # POST /evaluate, GET /evaluate/{run_id}
    deps.py                 # REST header auth (API_KEY/EVAL_KEY) + WS query-token auth
  /core
    config.py               # Pydantic BaseSettings, all tunables incl. MAX_TOTAL_EVIDENCE
    events.py                # TelemetryEvent envelope + event_type enum (incl. SESSION_RESYNC)
    embeddings.py             # bge-small load + embed() wrapper, LRU cache
    slots.py                  # NEW: loads data/corpus/<corpus_id>/slots.yaml -> SlotSchema
  /controller
    retrieval_controller.py     # WAIT/RETRIEVE/NO_RETRIEVAL state machine
    entity_extraction.py         # spaCy Matcher/EntityRuler against active SlotSchema
    suppression.py                 # presentation-only pattern match + first-turn guard
  /decomposition
    multi_intent.py           # spaCy dependency-parse compound-signal check + LLM decomposition + merge
  /retrieval
    dense.py                   # Qdrant client wrapper, corpus_id payload filter
    sparse_bm25.py               # BM25 index build/load/search, per-corpus_id pickle
    hybrid.py                     # concurrent dense+sparse per sub-query
    fusion.py                      # RRF + dedup + contradiction flagging
  /reranking
    cross_encoder.py            # ms-marco-MiniLM-L-6-v2 wrapper + cap_global_evidence
  /session
    session_store.py             # SessionState CRUD, in-memory + SQLite mirror, TTL sweep
    delta_engine.py                # refinement classification + delta retrieval
  /generation
    prompt_builder.py             # evidence + session state -> LLM prompt
    streaming_generator.py          # sentence-boundary streaming + inline grounding
  /grounding
    citation_validator.py         # claim segmentation, id-membership, entailment, always-both contradiction
  /telemetry
    event_logger.py               # async queue -> batched SQLite writer
    metrics.py                      # 10 metric formulas from TelemetryEvent log
  /models
    document.py, chunk.py, embedding.py, retrieval_event.py, sub_query.py,
    evidence.py, citation.py, session_state.py, answer_version.py,
    telemetry_event.py, evaluation_result.py
  main.py                       # FastAPI app factory, router mounting, async startup ingestion, /health, /ready
/tests
  unit/                         # one file per module, test_<module>.py
  integration/                  # test_full_pipeline.py, test_session_isolation.py,
                                 # test_refinement_flow.py, test_suppression.py, test_corpus_isolation.py
/benchmarks
  streaming_suite_v1/            # 10 test-case fixtures, gold labels (private/held-out)
  harness.py                      # replays fixtures through the live API, scores metrics
  results/                         # per-run JSON output
/scripts
  ingest_corpus.py                 # chunk, embed, index (Qdrant + BM25), idempotent by content_hash, --corpus-id
  run_benchmark.py                  # CLI wrapper around benchmarks/harness.py
  seed_data.py                       # optional demo corpus for local smoke-testing
/data
  corpus/<corpus_id>/               # raw supplied documents + slots.yaml (read-only input)
  processed/                          # chunks__<corpus_id>.jsonl, bm25__<corpus_id>.pkl, manifest__<corpus_id>.json (generated, gitignored)
/docker
  Dockerfile                          # CMD is uvicorn only; ingestion runs in-process at startup
  docker-compose.yml
/docs
  ARCHITECTURE.md
  API.md
  TELEMETRY.md
  EVALUATION.md
.github/workflows/ci.yml               # NEW: ruff -> pytest -> benchmark suite
.env.example
pyproject.toml                          # [tool.ruff] config
requirements.lock
README.md
PRD_TRD.md                               # this file
```

## 10. Implementation roadmap

Detailed 10-phase plan with files/tests/acceptance criteria per phase: unchanged from the original TRD Part 10, with the insertions below. The Theme 4 Guide's own roadmap (§7 of the PDF) names 5 macro-phases; this project's 10 phases are a finer-grained breakdown of those 5, mapped here for traceability:

| Guide macro-phase | This spec's phases |
|---|---|
| Phase 1: Framing & Foundation | Phase 1 (Foundation), Phase 2 (Corpus + Retrieval) |
| Phase 2: Controller & Live Stream Simulation | Phase 3 (Streaming Controller) |
| Phase 3: Multi-Intent Parsing & Evidence Fusion | Phase 4 (Multi-Intent), Phase 5 (Evidence Fusion) |
| Phase 4: Session Refinement & State Management | Phase 6 (Session Refinement), Phase 7 (Grounding + Citations) |
| Phase 5: Telemetry, Benchmarking & Packaging | Phase 8 (Telemetry), Phase 9 (Evaluation), Phase 10 (Docker + Deployment) |

Insertions from this freeze:
- **Phase 1** additionally scaffolds: spaCy model download, `app/core/slots.py`, `pyproject.toml` `[tool.ruff]`, `.github/workflows/ci.yml` skeleton.
- **Phase 2** additionally: `ingest_corpus.py --corpus-id`, `slots.yaml` validation, per-`corpus_id` BM25 pickle, `corpus_id` Qdrant payload field.
- **Phase 3**: `entity_extraction.py` implemented against spaCy + `SlotSchema`; `suppression.py` gets the first-turn guard.
- **Phase 4**: `has_compound_signal` implemented via spaCy dependency parse.
- **Phase 5**: reranker gains `cap_global_evidence`.
- **Phase 7**: contradiction handling simplified to always-both.
- **Phase 8**: event enum gains `SESSION_RESYNC`; `RetrievalEvent.trigger` gains `multi_intent`.
- **Phase 10**: Dockerfile `CMD` simplified to `uvicorn` only; ingestion moves to FastAPI startup event; CI workflow runs ruff → pytest → benchmark.

## 11. Risk register

Unchanged from the original TRD Part 11 (14 risks, causes, detections, mitigations, tests) — no risk is invalidated or newly introduced by this freeze. One risk is *reduced* by REQ-DEPLOY-01 ("Deployment failure — missing env var, corpus not mounted" is now also detected earlier via `/ready` never turning healthy, rather than only via a hard container-start failure).

## 12. Final implementation contract

- **Stack**: §8 above.
- **Architecture**: one FastAPI process, five deterministic pipeline stages, no agent orchestration, per-session `asyncio.Lock` concurrency, async telemetry queue. Full brief: `docs/ARCHITECTURE.md`.
- **APIs**: full contract in `docs/API.md`.
- **Telemetry**: full event catalog and metric formulas in `docs/TELEMETRY.md`.
- **Evaluation**: full G1–G6 gate definitions, REQ-ID↔test↔gate mapping, ablation requirements, submission deliverables in `docs/EVALUATION.md`.
- **Data models**: §7 above.
- **Repository structure**: §9 above.
- **Deployment command**: `docker compose up --build`, then `docker compose exec app python scripts/run_benchmark.py --test-set benchmarks/streaming_suite_v1` to verify G1–G6.

This specification is frozen. Any further change must be logged in §0 (Change log) with the same A–E resolution format used for the original 10 items.
