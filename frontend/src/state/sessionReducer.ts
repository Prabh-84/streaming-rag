import type {
  AnswerVersionCreatedPayload,
  CitationCreatedPayload,
  ErrorPayload,
  RerankCompletedPayload,
  RetrievalCompletedPayload,
  RetrievalDecisionPayload,
  RetrievalStartedPayload,
  SessionResyncPayload,
  SubqueryCreatedPayload,
  TelemetryEvent,
  UncertaintyPayload,
} from "../api/types";
import type { ConnectionState } from "../api/wsClient";
import { describeEvent } from "./activityLabels";
import { AppSessionState, StageId, TranscriptChunkState } from "./types";

export type SessionAction =
  | { type: "CONNECTION_STATE"; state: ConnectionState }
  | { type: "CHUNK_SENT"; chunk: TranscriptChunkState }
  | { type: "EVENT"; event: TelemetryEvent }
  | { type: "DISMISS_RESYNC_NOTICE" }
  | { type: "DISMISS_ERROR" };

function setStage(
  state: AppSessionState,
  stage: StageId,
  status: AppSessionState["pipeline"]["stages"][StageId],
): AppSessionState {
  return { ...state, pipeline: { ...state.pipeline, stages: { ...state.pipeline.stages, [stage]: status } } };
}

function startNewTurn(state: AppSessionState, traceId: string): AppSessionState {
  return {
    ...state,
    currentTraceId: traceId,
    subqueries: [],
    evidence: [],
    answer: { versionNo: null, text: "", citationChunkIds: [], isStreaming: false, supersedes: null },
    pipeline: {
      lastDecision: state.pipeline.lastDecision,
      stages: {
        TRANSCRIPT: "completed",
        CONTROLLER: "completed",
        MULTI_INTENT: "active",
        RETRIEVAL: "idle",
        RERANK: "idle",
        GROUND: "idle",
        ANSWER: "idle",
      },
    },
  };
}

function appendActivity(state: AppSessionState, event: TelemetryEvent): AppSessionState {
  const { label, detail } = describeEvent(event);
  const item = { id: event.event_id, timestamp: event.timestamp, eventType: event.event_type, label, detail, raw: event };
  // Newest first, capped so a long demo session doesn't grow the DOM unbounded.
  return { ...state, activity: [item, ...state.activity].slice(0, 300) };
}

function allSubqueriesRetrieved(state: AppSessionState): boolean {
  if (state.subqueries.length === 0) return false;
  return state.subqueries.every((s) => s.retrievalCompleted.dense && s.retrievalCompleted.sparse);
}

function allSubqueriesReranked(state: AppSessionState): boolean {
  if (state.subqueries.length === 0) return false;
  return state.subqueries.every((s) => s.rerankedChunkIds.length > 0 || s.uncertainty);
}

export function sessionReducer(state: AppSessionState, action: SessionAction): AppSessionState {
  switch (action.type) {
    case "CONNECTION_STATE":
      return { ...state, connection: action.state };

    case "CHUNK_SENT": {
      const transcript = [...state.transcript, action.chunk];
      let next = { ...state, transcript };
      if (state.pipeline.stages.TRANSCRIPT === "idle" || state.pipeline.stages.TRANSCRIPT === "completed") {
        next = setStage(next, "TRANSCRIPT", "active");
      }
      if (next.pipeline.stages.CONTROLLER === "idle" || next.pipeline.stages.CONTROLLER === "completed") {
        next = setStage(next, "CONTROLLER", "waiting");
      }
      return next;
    }

    case "DISMISS_RESYNC_NOTICE":
      return { ...state, resyncNotice: null };

    case "DISMISS_ERROR":
      return { ...state, lastError: null };

    case "EVENT":
      return applyEvent(state, action.event);

    default:
      return state;
  }
}

function applyEvent(state: AppSessionState, event: TelemetryEvent): AppSessionState {
  let next = appendActivity(state, event);

  switch (event.event_type) {
    case "RETRIEVAL_DECISION": {
      const p = event.payload as RetrievalDecisionPayload;
      next = { ...next, pipeline: { ...next.pipeline, lastDecision: p } };
      if (p.decision === "RETRIEVE") {
        next = startNewTurn(next, event.trace_id);
      } else if (p.decision === "WAIT") {
        next = setStage(next, "CONTROLLER", "waiting");
      } else {
        next = setStage(next, "CONTROLLER", "completed");
      }
      return next;
    }

    case "SUBQUERY_CREATED": {
      const p = event.payload as SubqueryCreatedPayload;
      const subqueries = [
        ...next.subqueries,
        {
          subQueryId: p.sub_query_id,
          text: p.text,
          intentLabel: p.intent_label,
          trigger: null,
          retrievalStarted: { dense: false, sparse: false },
          retrievalCompleted: {},
          rerankedChunkIds: [],
          rerankScores: [],
          uncertainty: null,
        },
      ];
      next = { ...next, subqueries };
      next = setStage(next, "MULTI_INTENT", "completed");
      next = setStage(next, "RETRIEVAL", "active");
      return next;
    }

    case "RETRIEVAL_STARTED": {
      const p = event.payload as RetrievalStartedPayload;
      const subqueries = next.subqueries.map((s) =>
        s.subQueryId === p.sub_query_id
          ? { ...s, trigger: p.trigger, retrievalStarted: { ...s.retrievalStarted, [p.mode]: true } }
          : s,
      );
      return setStage({ ...next, subqueries }, "RETRIEVAL", "active");
    }

    case "RETRIEVAL_COMPLETED": {
      const p = event.payload as RetrievalCompletedPayload;
      const subqueries = next.subqueries.map((s) =>
        s.subQueryId === p.sub_query_id
          ? {
              ...s,
              retrievalCompleted: {
                ...s.retrievalCompleted,
                [p.mode]: { resultCount: p.result_count, latencyMs: p.latency_ms },
              },
            }
          : s,
      );
      next = { ...next, subqueries };
      if (allSubqueriesRetrieved(next)) {
        next = setStage(next, "RETRIEVAL", "completed");
        next = setStage(next, "RERANK", "active");
      }
      return next;
    }

    case "RERANK_COMPLETED": {
      const p = event.payload as RerankCompletedPayload;
      const subqueries = next.subqueries.map((s) =>
        s.subQueryId === p.sub_query_id
          ? { ...s, rerankedChunkIds: p.ranked_chunk_ids, rerankScores: p.scores }
          : s,
      );
      const newEvidence = p.ranked_chunk_ids.map((chunkId, i) => ({
        chunkId,
        subQueryId: p.sub_query_id,
        rerankScore: p.scores[i],
        docId: null,
        section: null,
        claimText: null,
      }));
      next = { ...next, subqueries, evidence: [...next.evidence, ...newEvidence] };
      if (allSubqueriesReranked(next)) {
        next = setStage(next, "RERANK", "completed");
        next = setStage(next, "GROUND", "active");
      }
      return next;
    }

    case "CITATION_CREATED": {
      const p = event.payload as CitationCreatedPayload;
      const evidence = next.evidence.map((e) =>
        e.chunkId === p.chunk_id
          ? { ...e, docId: p.doc_id, section: p.section, claimText: p.claim_text }
          : e,
      );
      return setStage({ ...next, evidence }, "GROUND", "active");
    }

    case "UNCERTAINTY": {
      const p = event.payload as UncertaintyPayload;
      const subqueries = next.subqueries.map((s) =>
        s.subQueryId === p.sub_query_id
          ? { ...s, uncertainty: { reason: p.reason, clarifyingQuestion: p.clarifying_question } }
          : s,
      );
      return setStage({ ...next, subqueries }, "GROUND", "active");
    }

    case "ANSWER_DELTA": {
      const p = event.payload as { version_no: number; text_delta: string; is_final_sentence: boolean };
      const answer = {
        ...next.answer,
        versionNo: p.version_no,
        text: next.answer.text ? `${next.answer.text} ${p.text_delta}` : p.text_delta,
        isStreaming: true,
      };
      return setStage({ ...next, answer }, "ANSWER", "active");
    }

    case "ANSWER_VERSION_CREATED": {
      const p = event.payload as AnswerVersionCreatedPayload;
      const answer = {
        versionNo: p.version_no,
        text: p.text,
        citationChunkIds: p.citations,
        isStreaming: false,
        supersedes: p.supersedes,
      };
      next = { ...next, answer };
      next = setStage(next, "GROUND", "completed");
      next = setStage(next, "ANSWER", "completed");
      return next;
    }

    case "SESSION_RESYNC": {
      const p = event.payload as SessionResyncPayload;
      return {
        ...next,
        entities: p.entities,
        resyncNotice: { latestAnswerVersion: p.latest_answer_version },
      };
    }

    case "ERROR": {
      const p = event.payload as ErrorPayload;
      const stageMap: Record<string, StageId> = {
        controller: "CONTROLLER",
        stream: "CONTROLLER",
        generation: "ANSWER",
      };
      const stage = stageMap[p.stage];
      next = { ...next, lastError: p };
      if (stage) next = setStage(next, stage, "failed");
      return next;
    }

    default:
      return next;
  }
}
