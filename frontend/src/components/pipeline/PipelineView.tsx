import { PanelHeader } from "../common/PanelHeader";
import { MultiIntentPanel } from "./MultiIntentPanel";
import { PipelineStage } from "./PipelineStage";
import type { AppSessionState } from "../../state/types";
import { STAGE_ORDER } from "../../state/types";
import styles from "./PipelineView.module.css";

const STAGE_LABEL: Record<(typeof STAGE_ORDER)[number], string> = {
  TRANSCRIPT: "Transcript",
  CONTROLLER: "Controller",
  MULTI_INTENT: "Multi-Intent",
  RETRIEVAL: "Retrieval",
  RERANK: "Rerank",
  GROUND: "Ground",
  ANSWER: "Answer",
};

function controllerDetail(state: AppSessionState): string | null {
  const d = state.pipeline.lastDecision;
  if (!d) return null;
  return d.trigger ? `${d.decision} · ${d.trigger} (${d.reason})` : `${d.decision} · ${d.reason}`;
}

function retrievalDetail(state: AppSessionState): string | null {
  if (state.subqueries.length === 0) return null;
  const completedCalls = state.subqueries.reduce(
    (n, s) => n + (s.retrievalCompleted.dense ? 1 : 0) + (s.retrievalCompleted.sparse ? 1 : 0),
    0,
  );
  return `${completedCalls}/${state.subqueries.length * 2} retrieval calls completed`;
}

function rerankDetail(state: AppSessionState): string | null {
  if (state.evidence.length === 0) return null;
  return `${state.evidence.length} evidence item${state.evidence.length === 1 ? "" : "s"} kept`;
}

function groundDetail(state: AppSessionState): string | null {
  const citations = state.evidence.filter((e) => e.claimText).length;
  const uncertain = state.subqueries.filter((s) => s.uncertainty).length;
  const parts: string[] = [];
  if (citations > 0) parts.push(`${citations} grounded`);
  if (uncertain > 0) parts.push(`${uncertain} uncertain`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

function answerDetail(state: AppSessionState): string | null {
  if (state.answer.versionNo == null) return null;
  return `Answer Version ${state.answer.versionNo}`;
}

const DETAIL_FN: Record<(typeof STAGE_ORDER)[number], (s: AppSessionState) => string | null> = {
  TRANSCRIPT: (s) => (s.transcript.length > 0 ? `${s.transcript.length} chunk(s) received` : null),
  CONTROLLER: controllerDetail,
  MULTI_INTENT: (s) => (s.subqueries.length > 0 ? `${s.subqueries.length} sub-quer${s.subqueries.length === 1 ? "y" : "ies"}` : null),
  RETRIEVAL: retrievalDetail,
  RERANK: rerankDetail,
  GROUND: groundDetail,
  ANSWER: answerDetail,
};

export function PipelineView({ state }: { state: AppSessionState }) {
  return (
    <div className={styles.panel}>
      <PanelHeader title="Live RAG Pipeline" />
      <div className={styles.stages}>
        {STAGE_ORDER.map((stageId) => (
          <div key={stageId}>
            <PipelineStage
              label={STAGE_LABEL[stageId]}
              status={state.pipeline.stages[stageId]}
              detail={DETAIL_FN[stageId](state)}
            />
            {stageId === "MULTI_INTENT" && <MultiIntentPanel subqueries={state.subqueries} />}
          </div>
        ))}
      </div>
    </div>
  );
}
