# Streaming Live RAG — Telemetry & Observability Schema

Canonical source for the event envelope, event catalog, writer architecture, and the 10 benchmark metric formulas. Referenced from `../PRD_TRD.md` §5.1/§12 and `EVALUATION.md`. This is also the deliverable named "Telemetry & Observability Schema" in the Theme 4 Guide's Engineering Deliverables Checklist.

## 1. Event envelope

Every telemetry event and every WebSocket server→client frame shares one envelope (REQ-OBS-01):

```json
{
  "event_id": "string, uuid",
  "session_id": "string",
  "timestamp": "ISO-8601 UTC",
  "event_type": "string, enum — see §2",
  "payload": {},
  "trace_id": "string, uuid, shared by one pipeline turn"
}
```

`trace_id` is generated once per turn (one user-chunk-triggered pipeline run, or one refinement pass) and threaded through every event that turn produces, so `GET /session/{id}/events?trace_id=...` reconstructs one full pipeline execution end-to-end.

## 2. Event types and payloads

| event_type | payload fields | Emitted by |
|---|---|---|
| `TRANSCRIPT_CHUNK` | `seq, text_delta, t_offset_ms, is_final` | Client (inbound) |
| `RETRIEVAL_DECISION` | `decision` (WAIT\|RETRIEVE\|NO_RETRIEVAL), `trigger`, `reason` | Retrieval Controller |
| `SUBQUERY_CREATED` | `sub_query_id, text, intent_label` | Multi-Intent Decomposer |
| `RETRIEVAL_STARTED` | `sub_query_id, mode` (dense\|sparse), `t_offset_ms` | Hybrid Retriever |
| `RETRIEVAL_COMPLETED` | `sub_query_id, mode, result_count, latency_ms` | Hybrid Retriever |
| `RERANK_COMPLETED` | `sub_query_id, ranked_chunk_ids, scores` | Reranker |
| `CITATION_CREATED` | `chunk_id, doc_id, section, claim_text` | Grounding Validator |
| `ANSWER_VERSION_CREATED` | `version_no, text, citations[], supersedes` | Session-Aware Synthesis |
| `ANSWER_DELTA` | `version_no, text_delta, is_final_sentence` | Streaming Generator |
| `UNCERTAINTY` | `sub_query_id, reason, clarifying_question` | Grounding Validator |
| `ERROR` | `stage, error_type, message, recoverable` | Any stage |
| `SESSION_UPDATED` | `field, old_value, new_value` | Session Store |
| `SESSION_RESYNC` **(NEW)** | `latest_answer_version, entities, last_seq` | API layer, on WebSocket reconnect to an existing session (REQ-STREAM-03) |

### 2.1 `RetrievalEvent.trigger` enum (persisted record backing `RETRIEVAL_STARTED`/`RETRIEVAL_COMPLETED`)

`provisional | final | delta | forced_after_max_wait | multi_intent`

`multi_intent` **(NEW, REQ-OBS-05)** tags a sub-query-level retrieval produced by decomposition (n≥2 sub-queries), distinct from the controller-level trigger (`provisional`/`final`/`delta`/`forced_after_max_wait`) that caused the *first* sub-query's retrieval to fire. This closes a gap against the Theme 4 Guide's example structured output record, which tags a decomposed sub-query's retrieval as `"trigger": "multi_intent"` — a value the original 4-item enum had no slot for.

Example (illustrates the corrected tagging, not gold benchmark data):
```json
{
  "retrieval_events": [
    {"timestamp_s": 0.8, "query": "Pune workshop venue capacity 30", "trigger": "provisional"},
    {"timestamp_s": 1.6, "query": "cancellation policy workshop venues Pune", "trigger": "multi_intent"},
    {"timestamp_s": 1.6, "query": "catering service options workshop Pune", "trigger": "multi_intent"}
  ]
}
```

## 3. `SESSION_RESYNC` payload detail

```json
{
  "event_type": "SESSION_RESYNC",
  "payload": {
    "latest_answer_version": 2,
    "entities": {"location": "Pune", "capacity": 30},
    "last_seq": 7
  }
}
```
Emitted exactly once, immediately after a WebSocket reconnect is accepted for an existing, non-expired `session_id`, before any further `TRANSCRIPT_CHUNK` is processed.

## 4. Writer architecture (REQ-OBS-04)

Events are written via an `asyncio.Queue` consumed by a background writer task that batches inserts to SQLite every 100ms. Emission never blocks the response/streaming path — the queue write itself must add no more than 5ms to the critical path per event. Every pipeline stage is wrapped so that an exception becomes an `ERROR` telemetry event (`{stage, error_type, message, recoverable}`) with a defined fallback (see `PRD_TRD.md` §11 risk register / original TRD §9.3), rather than propagating a raw failure to the WebSocket.

## 5. Metric formulas (computed only from the TelemetryEvent log — never from re-parsing generated text)

```text
Early Retrieval Rate =
    count(eligible queries where RETRIEVAL_STARTED.t_offset_ms < utterance_end_offset_ms)
    / count(eligible queries)
    # "eligible" excludes cases with gold label NO_RETRIEVAL

Multi-Intent Accuracy =
    count(compound queries where predicted_sub_intents == gold_sub_intents, set match)
    / count(compound queries)

Citation Grounding Rate =
    count(sampled factual assertions with a valid, entailed citation)
    / count(sampled factual assertions)

Fabricated Citation Rate =
    count(citations whose chunk_id is absent from that turn's evidence set)
    / count(citations emitted)
    # gate requires this to equal 0

Retrieval Precision =
    count(retrieved chunks marked relevant in gold) / count(retrieved chunks)

Retrieval Recall =
    count(retrieved chunks marked relevant in gold) / count(gold-relevant chunks)

Time-to-First-Token (TTFT) =
    timestamp(first ANSWER_DELTA) - timestamp(RETRIEVE decision OR utterance_end, whichever applies)

End-to-End Latency =
    timestamp(ANSWER_VERSION_CREATED) - timestamp(utterance_end)

Token Cost =
    sum(prompt_tokens + completion_tokens across all LLM calls in the turn) * provider_rate

Session Refinement Accuracy =
    count(late-constraint cases where carried_forward claims are unchanged
          AND only delta sub-queries were issued AND no session state was cleared)
    / count(late-constraint cases)
```

All ten are computed by `app/telemetry/metrics.py` directly from the `TelemetryEvent` log, so scoring is deterministic and independent of any particular LLM's phrasing. Gate thresholds and pass/fail mapping: `EVALUATION.md` §1.

## 6. Coverage requirement (Gate G6)

REQ-OBS-02: 100% of sessions must have a complete, replayable trace covering transcript chunks, retrieval decisions and triggers, sub-queries, retrieved chunk IDs and scores, rerank order, citations, answer versions, token usage, latency per stage, uncertainty flags, and errors. Verified by `test_event_envelope_complete` and `test_trace_id_threading` (`PRD_TRD.md` §4.5/§5.1).
