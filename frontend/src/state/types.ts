import type {
  ErrorPayload,
  EventType,
  TelemetryEvent,
} from "../api/types";
import type { ConnectionState } from "../api/wsClient";

export type StageId =
  | "TRANSCRIPT"
  | "CONTROLLER"
  | "MULTI_INTENT"
  | "RETRIEVAL"
  | "RERANK"
  | "GROUND"
  | "ANSWER";

export const STAGE_ORDER: StageId[] = [
  "TRANSCRIPT",
  "CONTROLLER",
  "MULTI_INTENT",
  "RETRIEVAL",
  "RERANK",
  "GROUND",
  "ANSWER",
];

export type StageStatus = "idle" | "waiting" | "active" | "completed" | "failed";

export interface PipelineState {
  stages: Record<StageId, StageStatus>;
  lastDecision: { decision: string; trigger: string | null; reason: string } | null;
}

export interface SubqueryRetrievalMode {
  resultCount: number;
  latencyMs: number;
}

export interface SubqueryState {
  subQueryId: string;
  text: string;
  intentLabel: string;
  trigger: string | null;
  retrievalStarted: { dense: boolean; sparse: boolean };
  retrievalCompleted: {
    dense?: SubqueryRetrievalMode;
    sparse?: SubqueryRetrievalMode;
  };
  rerankedChunkIds: string[];
  rerankScores: number[];
  uncertainty: { reason: string; clarifyingQuestion: string | null } | null;
}

export interface EvidenceItem {
  chunkId: string;
  subQueryId: string;
  rerankScore: number;
  docId: string | null;
  section: string | null;
  claimText: string | null;
}

export interface AnswerState {
  versionNo: number | null;
  text: string;
  citationChunkIds: string[];
  isStreaming: boolean;
  supersedes: number | null;
}

export interface ActivityItem {
  id: string;
  timestamp: string;
  eventType: EventType;
  label: string;
  detail: string | null;
  raw: TelemetryEvent;
}

export interface TranscriptChunkState {
  seq: number;
  textDelta: string;
  tOffsetMs: number;
  isFinal: boolean;
  sentAt: number;
}

export interface AppSessionState {
  connection: ConnectionState;
  currentTraceId: string | null;
  pipeline: PipelineState;
  subqueries: SubqueryState[];
  evidence: EvidenceItem[];
  answer: AnswerState;
  answerHistory: AnswerState[];
  activity: ActivityItem[];
  transcript: TranscriptChunkState[];
  entities: Record<string, string>;
  resyncNotice: { latestAnswerVersion: number | null } | null;
  lastError: ErrorPayload | null;
}

export function createInitialSessionState(): AppSessionState {
  return {
    connection: "disconnected",
    currentTraceId: null,
    pipeline: {
      stages: {
        TRANSCRIPT: "idle",
        CONTROLLER: "idle",
        MULTI_INTENT: "idle",
        RETRIEVAL: "idle",
        RERANK: "idle",
        GROUND: "idle",
        ANSWER: "idle",
      },
      lastDecision: null,
    },
    subqueries: [],
    evidence: [],
    answer: { versionNo: null, text: "", citationChunkIds: [], isStreaming: false, supersedes: null },
    answerHistory: [],
    activity: [],
    transcript: [],
    entities: {},
    resyncNotice: null,
    lastError: null,
  };
}
