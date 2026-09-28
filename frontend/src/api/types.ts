/**
 * Types mirror the actual backend contracts, verified directly against source:
 * app/api/session.py, app/api/stream.py, app/core/events.py, app/models/transcript_chunk.py,
 * docs/API.md, docs/TELEMETRY.md. Nothing here is invented - every field name/shape below is
 * copied from what the backend actually sends, not guessed.
 */

// ---- REST: POST /session ----------------------------------------------------------------------

export interface CreateSessionRequest {
  corpus_id?: string;
  client_meta?: Record<string, unknown>;
}

export interface CreateSessionResponse {
  session_id: string;
  created_at: string;
  ws_url: string; // relative path + query token, as returned by the backend - never construct this ourselves
}

// ---- REST: GET /session/{id} --------------------------------------------------------------------

export interface SessionSnapshot {
  session_id: string;
  corpus_id: string;
  status: "active" | "closed" | "expired";
  entities: Record<string, string>;
  latest_answer_version: number | null;
  created_at: string;
  last_active_at: string;
}

// ---- WebSocket outbound: TRANSCRIPT_CHUNK -------------------------------------------------------

export interface TranscriptChunkPayload {
  seq: number;
  text_delta: string;
  t_offset_ms: number;
  is_final: boolean;
}

export interface TranscriptChunkFrame {
  event_type: "TRANSCRIPT_CHUNK";
  payload: TranscriptChunkPayload;
}

// ---- WebSocket inbound: the standard TelemetryEvent envelope (docs/TELEMETRY.md §1) -------------

export type EventType =
  | "RETRIEVAL_DECISION"
  | "SUBQUERY_CREATED"
  | "RETRIEVAL_STARTED"
  | "RETRIEVAL_COMPLETED"
  | "RERANK_COMPLETED"
  | "CITATION_CREATED"
  | "ANSWER_VERSION_CREATED"
  | "ANSWER_DELTA"
  | "UNCERTAINTY"
  | "ERROR"
  | "SESSION_UPDATED"
  | "SESSION_RESYNC";

export interface RetrievalDecisionPayload {
  decision: "WAIT" | "RETRIEVE" | "NO_RETRIEVAL";
  trigger: "provisional" | "final" | "delta" | "forced_after_max_wait" | null;
  reason: string;
}

export interface SubqueryCreatedPayload {
  sub_query_id: string;
  text: string;
  intent_label: string;
}

export interface RetrievalStartedPayload {
  sub_query_id: string;
  mode: "dense" | "sparse";
  t_offset_ms: number;
  trigger: string;
}

export interface RetrievalCompletedPayload {
  sub_query_id: string;
  mode: "dense" | "sparse";
  result_count: number;
  latency_ms: number;
  result_chunk_ids: string[];
  scores: number[];
  t_offset_ms: number;
  trigger: string;
}

export interface RerankCompletedPayload {
  sub_query_id: string;
  ranked_chunk_ids: string[];
  scores: number[];
}

export interface CitationCreatedPayload {
  chunk_id: string;
  doc_id: string;
  section: string;
  claim_text: string;
}

export interface AnswerVersionCreatedPayload {
  version_no: number;
  text: string;
  citations: string[];
  supersedes: number | null;
}

export interface AnswerDeltaPayload {
  version_no: number;
  text_delta: string;
  is_final_sentence: boolean;
}

export interface UncertaintyPayload {
  sub_query_id: string;
  reason: string;
  clarifying_question: string | null;
}

export interface ErrorPayload {
  stage: string;
  error_type: string;
  message: string;
  recoverable: boolean;
}

export interface SessionResyncPayload {
  latest_answer_version: number | null;
  entities: Record<string, string>;
  last_seq: number | null;
}

export interface SessionUpdatedPayload {
  field: string;
  old_value: unknown;
  new_value: unknown;
}

export type EventPayload =
  | RetrievalDecisionPayload
  | SubqueryCreatedPayload
  | RetrievalStartedPayload
  | RetrievalCompletedPayload
  | RerankCompletedPayload
  | CitationCreatedPayload
  | AnswerVersionCreatedPayload
  | AnswerDeltaPayload
  | UncertaintyPayload
  | ErrorPayload
  | SessionResyncPayload
  | SessionUpdatedPayload;

export interface TelemetryEvent<P extends EventPayload = EventPayload> {
  event_id: string;
  session_id: string;
  timestamp: string;
  event_type: EventType;
  payload: P;
  trace_id: string;
}

// WS close codes the backend actually uses (app/api/stream.py)
export const WS_AUTH_FAILED = 4401;
export const WS_UNKNOWN_SESSION = 4404;
