/** Mirrors app/grounding/citation_validator.py's own _CITATION_TAG regex exactly, so a tag the
 * frontend highlights is guaranteed to be the same shape the backend validator recognizes. */
const CITATION_TAG = /\[\s*(\S+)\s+(\S+)\s*\]/g;

export interface AnswerSegment {
  kind: "text" | "citation";
  text: string;
  docId?: string;
  section?: string;
}

export function splitAnswerIntoSegments(text: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let lastIndex = 0;
  for (const match of text.matchAll(CITATION_TAG)) {
    const [full, docId, section] = match;
    const start = match.index ?? 0;
    if (start > lastIndex) segments.push({ kind: "text", text: text.slice(lastIndex, start) });
    segments.push({ kind: "citation", text: full, docId, section });
    lastIndex = start + full.length;
  }
  if (lastIndex < text.length) segments.push({ kind: "text", text: text.slice(lastIndex) });
  return segments;
}
