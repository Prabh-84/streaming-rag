/**
 * The primary demo transcript, validated against the real backend and the real
 * northstar_demo_extended corpus (see PRD_TRD.md-adjacent validation notes). This is INPUT data
 * only - a sequence of TRANSCRIPT_CHUNK frames the frontend sends through the real WebSocket API.
 * Every downstream event (decisions, sub-queries, retrieval, reranking, citations, the answer
 * itself) is produced live by the real backend; nothing about the response is scripted here.
 */
export interface DemoChunk {
  textDelta: string;
  tOffsetMs: number;
  isFinal: boolean;
  /** Wall-clock delay before sending this chunk, to make the "streaming" effect visible. */
  sendDelayMs: number;
}

const BASE = "I have a question about student policies at Northstar";
const COMPOUND = `${BASE}, specifically the library fine policy and the hostel visitor registration procedure`;

export const NORTHSTAR_DEMO_SCRIPT: DemoChunk[] = [
  { textDelta: BASE, tOffsetMs: 0, isFinal: false, sendDelayMs: 0 },
  { textDelta: COMPOUND, tOffsetMs: 900, isFinal: false, sendDelayMs: 900 },
  { textDelta: `${COMPOUND}.`, tOffsetMs: 1800, isFinal: true, sendDelayMs: 900 },
];

export const DEMO_QUERY_LABEL =
  "Tell me about the library fine policy and the hostel visitor registration procedure.";
