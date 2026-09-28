import type { TelemetryEvent } from "../api/types";

/** Human-readable labels for the activity panel. Every number/name shown is read directly off
 * the real event payload - nothing here invents a count or a value. */
export function describeEvent(event: TelemetryEvent): { label: string; detail: string | null } {
  switch (event.event_type) {
    case "RETRIEVAL_DECISION": {
      const p = event.payload as { decision: string; trigger: string | null; reason: string };
      if (p.decision === "WAIT") return { label: "Waiting for more context", detail: p.reason };
      if (p.decision === "NO_RETRIEVAL")
        return { label: "No retrieval needed", detail: p.reason };
      return { label: "Retrieval triggered", detail: `${p.trigger ?? p.reason}` };
    }
    case "SUBQUERY_CREATED": {
      const p = event.payload as { text: string; intent_label: string };
      return { label: "Sub-query created", detail: `${p.intent_label}: "${p.text}"` };
    }
    case "RETRIEVAL_STARTED": {
      const p = event.payload as { mode: string };
      return { label: `${p.mode === "dense" ? "Dense" : "Sparse"} retrieval started`, detail: null };
    }
    case "RETRIEVAL_COMPLETED": {
      const p = event.payload as { mode: string; result_count: number; latency_ms: number };
      return {
        label: "Retrieval completed",
        detail: `${p.result_count} candidates (${p.mode}, ${p.latency_ms}ms)`,
      };
    }
    case "RERANK_COMPLETED": {
      const p = event.payload as { ranked_chunk_ids: string[] };
      return { label: "Evidence reranked", detail: `${p.ranked_chunk_ids.length} kept` };
    }
    case "CITATION_CREATED": {
      const p = event.payload as { doc_id: string; section: string };
      return { label: "Citation created", detail: `${p.doc_id} ${p.section}` };
    }
    case "ANSWER_DELTA":
      return { label: "Answer streaming", detail: null };
    case "ANSWER_VERSION_CREATED": {
      const p = event.payload as { version_no: number; citations: string[] };
      return {
        label: "Grounded answer generated",
        detail: `version ${p.version_no}, ${p.citations.length} citation(s)`,
      };
    }
    case "UNCERTAINTY": {
      const p = event.payload as { reason: string };
      return { label: "Uncertainty flagged", detail: p.reason };
    }
    case "ERROR": {
      const p = event.payload as { stage: string; error_type: string };
      return { label: `Error in ${p.stage}`, detail: p.error_type };
    }
    case "SESSION_RESYNC":
      return { label: "Session resynchronized", detail: null };
    case "SESSION_UPDATED": {
      const p = event.payload as { field: string; new_value: unknown };
      return { label: "Session updated", detail: `${p.field} -> ${String(p.new_value)}` };
    }
    default:
      return { label: event.event_type, detail: null };
  }
}
