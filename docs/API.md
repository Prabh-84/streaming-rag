# Streaming Live RAG — API Contract

Canonical source for all endpoint, request/response, and error contracts. Referenced from `../PRD_TRD.md` §5/§12 — do not duplicate these schemas elsewhere; amend here only.

## 1. Authentication

- **REST endpoints** (`POST /session`, `GET /session/{id}`, `GET /session/{id}/events`, `POST /evaluate`, `GET /evaluate/{run_id}`): bearer token via the `Authorization` header — `Authorization: Bearer <API_KEY>`. `/evaluate` and `GET /evaluate/{run_id}` additionally require `EVAL_KEY` via a separate header, `X-Eval-Key: <EVAL_KEY>`. (REQ-SEC-04)
- **WebSocket** (`WS /session/{id}/stream`): browsers cannot set custom headers on a WS handshake, so the token is passed as a query parameter: `wss://.../session/{id}/stream?token=<API_KEY>`, validated identically to the header-based check. (REQ-SEC-06)
  - Residual note: query strings can appear in reverse-proxy/access logs. Mitigate at the infra layer (exclude this path from query-string logging) — this is an accepted, standard tradeoff for this exact browser limitation, not a weaker auth mechanism.

## 2. `POST /session`

Creates a new, empty session.

**Request**
```json
{"corpus_id": "default", "client_meta": {}}
```
Both fields optional; `corpus_id` defaults to `"default"`.

**Response `201`**
```json
{
  "session_id": "sess_9f2a",
  "created_at": "2026-09-23T10:00:00Z",
  "ws_url": "/session/sess_9f2a/stream?token=<API_KEY>"
}
```

**Errors**
- `401` — missing/invalid `API_KEY`.
- `400` — unknown `corpus_id` (not previously ingested — REQ-CORPUS-02).

## 3. `WS /session/{id}/stream`

Bidirectional. Auth: query-token (§1). Client sends `TRANSCRIPT_CHUNK` frames; server streams every event type in `TELEMETRY.md` §2.

**Client → server**
```json
{"event_type": "TRANSCRIPT_CHUNK", "payload": {"seq": 3, "text_delta": "...for 30 people, and I need...", "t_offset_ms": 800, "is_final": false}}
```

**Server → client (example: RETRIEVAL_DECISION)**
```json
{"event_id": "evt_01", "session_id": "sess_9f2a", "timestamp": "2026-09-23T10:00:00.812Z", "event_type": "RETRIEVAL_DECISION", "payload": {"decision": "RETRIEVE", "trigger": "provisional", "reason": "stable entities: venue, capacity=30"}, "trace_id": "trc_77"}
```

**Reconnection (REQ-STREAM-03)** — reconnecting to the same URL for an existing, non-expired `session_id` causes the server to immediately emit one `SESSION_RESYNC` event (payload: latest `AnswerVersion` + current entity/buffer state) before accepting further `TRANSCRIPT_CHUNK` frames. The client does not replay prior chunks; it resumes sending only new ones.

**Errors**
- Connection closes with code `4401` on auth failure (missing/invalid token).
- Connection closes with code `4404` on unknown or expired `session_id` (no resync attempted).
- A stage-level pipeline failure is delivered as an `ERROR` event on the open socket, never a socket close.

## 4. Standard event envelope

```json
{
  "event_id": "string, uuid",
  "session_id": "string",
  "timestamp": "ISO-8601 UTC",
  "event_type": "string, enum — see TELEMETRY.md §2",
  "payload": {},
  "trace_id": "string, uuid, shared by one pipeline turn"
}
```

Full event-type catalog and payload fields: `TELEMETRY.md` §2. This file only documents the endpoints that carry the envelope, not the envelope's contents.

## 5. `GET /session/{id}`

Current session snapshot.

**Response `200`**
```json
{"session_id": "sess_9f2a", "corpus_id": "default", "status": "active", "entities": {}, "latest_answer_version": 2, "created_at": "...", "last_active_at": "..."}
```

`corpus_id` reflects the value fixed at session creation (PRD_TRD.md §7.8) — immutable for the session's lifetime.

**Errors** — `404` unknown session.

## 6. `GET /session/{id}/events`

Full ordered event log for replay/audit (REQ-OBS-03). Query params: `trace_id` (filter to one pipeline turn), `event_type`, `since` (ISO timestamp). Transport: SSE stream of the envelope in §4, or `?format=json` for a single array response (used by the benchmark harness).

**Errors**
- `404` — unknown session.
- `410` — session's log purged post-TTL.

## 7. `POST /evaluate`

Triggers a benchmark run against a named, held-out test set. Requires `EVAL_KEY` (§1). Gated by an IP allowlist in production config (cost-incurring bulk pipeline runs).

**Request**
```json
{"test_set": "benchmarks/streaming_suite_v1", "corpus_id": "default"}
```

**Response `202`**
```json
{"run_id": "run_44", "status": "queued"}
```

## 8. `GET /evaluate/{run_id}`

Result of a triggered run (not polled synchronously — poll this endpoint).

**Response `200`**
```json
{
  "run_id": "run_44",
  "status": "complete",
  "gates": {"G1": "pass", "G2": {"value": 0.83, "target": 0.80, "pass": true}, "...": "..."},
  "metrics": {}
}
```

**Errors**
- `401`/`403` — bad or missing `EVAL_KEY`.
- `409` — a run is already in progress for that `test_set`.

Exact gate definitions, metric formulas, and REQ-ID mapping: `EVALUATION.md`.

## 9. Health and readiness (REQ-DEPLOY-01)

- `GET /health` — liveness. Returns `200` immediately, no dependency checks, never blocked by corpus ingestion.
- `GET /ready` — readiness. Returns `503` until Qdrant connectivity is confirmed, corpus ingestion has completed at least once, AND the embedding model has finished warming up (loaded once at startup, PRD_TRD.md §5.4 REQ-DEPLOY-01) — response body: `{"status": "ready"|"not_ready", "qdrant": bool, "ingestion": bool, "embedder": bool}`. Orchestrators/compose health probes intended to gate real traffic must target this endpoint, not `/health`.

Neither endpoint requires authentication (used by infrastructure health checks).
