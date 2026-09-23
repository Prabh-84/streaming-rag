# Streaming Live RAG — System Architecture Brief

Condensed from `../PRD_TRD.md`. Canonical detail lives there and in `API.md`, `TELEMETRY.md`, `EVALUATION.md` — this brief is the standalone submission document (Theme 4 Guide deliverable: "System Architecture Brief, ≤6 pages").

## 1. Overview

Streaming Live RAG answers a user's request while it is still arriving. It is one FastAPI process running five deterministic pipeline stages per conversational turn — no agent orchestration, no multi-service fan-out beyond the vector store. The engine: (1) decides per transcript chunk whether to retrieve, (2) decomposes compound requests into independent sub-queries, (3) retrieves from a supplied corpus only (dense + sparse, fused and reranked), (4) synthesizes a grounded, cited answer while streaming it sentence-by-sentence, and (5) refines that answer in place when the user adds a late constraint, instead of restarting.

## 2. Architecture diagram

```text
Client: transcript chunks
   │  TRANSCRIPT_CHUNK (WS, query-token auth)
   ▼
Retrieval Controller ──WAIT──▶ (loop)
   │
   ├──NO_RETRIEVAL──▶ Session-Aware Synthesis
   │
   └──RETRIEVE──▶ Multi-Intent Decomposer
                       │  SubQuery x N (parallel)
                       ▼
                  Hybrid Retriever (dense + sparse, per corpus_id)
                       ▼
                  Evidence Fusion (RRF + dedup + contradiction flag)
                       ▼
                  Cross-Encoder Reranker (+ global evidence cap, ≤15)
                       ▼
                  Grounding Validator
                       ▼
                  Session-Aware Synthesis ◀──▶ Session Store (SQLite-mirrored dict)
                       ▼
                  Streaming Answer + Citations (ANSWER_DELTA, sentence-by-sentence)

Every stage ──▶ Telemetry Event Log (async queue, batched SQLite writer)
Hybrid Retriever ◀──▶ Vector Store (Qdrant, corpus_id-filtered)
Hybrid Retriever ◀──▶ BM25 Sparse Index (per-corpus_id pickle)
```

## 3. Core pipeline components

| # | Component | Responsibility | Core engineering challenge |
|---|---|---|---|
| 1 | Retrieval Controller | Per-chunk WAIT/RETRIEVE/NO_RETRIEVAL decision from embedding stability + entity extraction | Balancing early-retrieval latency gains against premature, noisy searches from incomplete thoughts |
| 2 | Multi-Intent Decomposer | Splits compound, unsegmented utterances into discrete, search-ready sub-queries | Extracting distinct orthogonal questions without losing context or over-fragmenting |
| 3 | Hybrid Retriever + Evidence Fusion + Reranker | Dense+sparse search per sub-query, RRF fusion, dedup, cross-encoder rerank, global evidence cap | Merging multi-source evidence without diluting context or introducing contradictions |
| 4 | Session-Aware Synthesis (incl. Delta Engine) | Applies late-arriving constraints directly onto existing answer state; grounds and streams | Tracking answer versions, mutating only affected claims, never re-running full retrieval |
| 5 | Grounding Validator | Claim segmentation, citation-to-chunk verification, uncertainty/contradiction flagging | Rejecting fabricated citations by construction, not by post-hoc detection |
| 6 | Telemetry | Async structured event emission for every decision, retrieval, citation, version, cost, latency | Zero-blocking instrumentation under sub-second streaming constraints |

Non-responsibilities are load-bearing: the Controller never searches the corpus; the Decomposer never decides *whether* to retrieve; Fusion never calls an LLM; the Reranker never fetches new candidates; the Grounding Validator never generates text. This keeps each stage independently testable and auditable (Hard Constraint HC-5, Architectural Parsimony).

## 4. WAIT / RETRIEVE / NO_RETRIEVAL state machine

```text
IDLE → WAIT (chunk received)
WAIT → WAIT              [cosine < 0.90 across last K chunks, or no actionable entity]
WAIT → NO_RETRIEVAL       [presentation-only pattern matched, AND session has ≥1 prior answer]
WAIT → PROVISIONAL_RETRIEVE   [stable entity ≥2 consecutive chunks]
PROVISIONAL_RETRIEVE → DECOMPOSING   [utterance_end OR compound marker]
NO_RETRIEVAL → SYNTHESIS
DECOMPOSING → RETRIEVING     [≥1 sub-query]
RETRIEVING → FUSING          [all results returned or timeout]
FUSING → SYNTHESIS
SYNTHESIS → STREAMING
STREAMING → IDLE             [answer_version emitted]
IDLE → REFINING              [new chunk classified as refinement of active topic]
REFINING → DECOMPOSING        [delta sub_queries only]
IDLE → [closed]
```

Guard: `WAIT → PROVISIONAL_RETRIEVE` requires **both** semantic stability *and* a concrete, slot-bound entity — stability alone never fires retrieval (prevents Pitfall 1, eager/premature retrieval). The `NO_RETRIEVAL` presentation-only transition additionally requires a non-empty `session.answer_versions` — a first-turn "repeat that" has nothing to repeat, so it falls through to normal intent evaluation instead (first-turn suppression guard).

## 5. Data flow (one turn)

```text
transcript buffer → chunk embedding → stability check
   → entity/intent extract (spaCy Matcher + dependency parse against corpus slots.yaml)
   → actionable? ──no (presentation-only)──▶ suppress
                  └─yes──▶ decompose (skip if no compound signal)
                              → parallel dense+sparse search (scoped by corpus_id)
                              → RRF fuse + dedup + contradiction flag
                              → cross-encoder rerank
                              → global evidence cap (≤15 total, ≥1 per sub-intent) + token-budget truncate
                              → prompt assembly (evidence + session state)
                              → stream generation
                              → sentence-level grounding check (citation-set membership + entailment)
                              → client: ANSWER_DELTA
                              → SessionState.claims update
```

## 6. Key design decisions from the specification freeze

| Decision | Rationale |
|---|---|
| spaCy + `en_core_web_sm` for entity/compound-signal extraction | Deterministic, local CPU, auditable rule-matching — no black-box NER, no external call, negligible latency (HC-5) |
| Per-corpus `slots.yaml` | Domain-slot vocabulary is corpus data, not code — keeps the system genuinely corpus-agnostic |
| REST: `Authorization` header; WS: query-string token | Browsers cannot set custom headers on a WS handshake; same `API_KEY` secret, different transport |
| `SESSION_RESYNC` event on reconnect | Lets a dropped client resume from server-side `SessionState` without replaying chunks |
| First-turn suppression guard | A presentation-only pattern with no prior answer must never short-circuit into an undefined state |
| `MAX_TOTAL_EVIDENCE=15`, min 1 chunk/sub-intent | Enforces the previously-implicit global cap; the floor prevents one dominant sub-query starving another's citations |
| Always surface both sides of a contradiction | No data-model field existed for "authoritative source"; the safer, always-both behavior is also what the benchmark rewards |
| `corpus_id` filtering (Qdrant payload + per-corpus BM25 file) | Multi-corpus support reuses the same mandatory-scoping pattern already required for session isolation |
| Ingestion in FastAPI startup event, not blocking Docker `CMD` | `/health` (liveness) must never depend on ingestion; `/ready` (readiness) already existed to gate real traffic |
| ruff for lint/format in CI | One dependency instead of three; satisfies the Definition of Done's lint requirement |

## 7. Common pitfalls and their mitigations

| Pitfall | Mitigation | Enforcing REQ-ID |
|---|---|---|
| Eager/premature retrieval on noise | Stability **and** actionable-entity conjunction required, never stability alone | REQ-CTRL-01/02 |
| Context loss on late constraints | Delta engine diffs entities and retrieves only the delta; never clears `SessionState` | REQ-SESS-01/02 |
| Citation hallucination | Validator rejects any `chunk_id` not in that turn's evidence set, by construction | REQ-GROUND-01 |
| Ignoring presentation-only turns | Suppression checked before the Controller's RETRIEVE path; guarded against the first-turn edge case | REQ-SUPPRESS-01/02 |
| Over-fragmenting sub-queries | Syntactic gate before any LLM call; embedding-similarity merge at cosine > 0.85 | REQ-INTENT-02 |

## 8. Roadmap summary

Ten phases, dependency-ordered, mapping to the Theme 4 Guide's five macro-phases (full detail: `PRD_TRD.md` §10): Foundation → Corpus+Retrieval → Streaming Controller → Multi-Intent → Evidence Fusion → Session Refinement → Grounding+Citations → Telemetry → Evaluation → Docker+Deployment. Phases 2–4 are independent and parallelizable once Phase 1 lands.

## 9. Deployment shape

Two Docker services (`app`, `qdrant`); BM25 and SQLite live inside the `app` container's volume. `docker compose up --build` starts everything; `/health` reports liveness immediately, `/ready` gates traffic behind Qdrant connectivity and completed ingestion. Production: same single image on one managed container platform; SQLite→Postgres and self-hosted→Cloud Qdrant are config-only swaps. No Kubernetes, no service mesh (HC-5).
