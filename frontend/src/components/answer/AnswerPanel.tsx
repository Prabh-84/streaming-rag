import { PanelHeader } from "../common/PanelHeader";
import { splitAnswerIntoSegments } from "./citationParsing";
import type { AnswerState } from "../../state/types";
import styles from "./AnswerPanel.module.css";

export function AnswerPanel({
  answer,
  onCitationClick,
}: {
  answer: AnswerState;
  onCitationClick: (docId: string, section: string) => void;
}) {
  const hasContent = answer.text.length > 0;
  const segments = hasContent ? splitAnswerIntoSegments(answer.text) : [];

  return (
    <div className={styles.panel}>
      <PanelHeader
        title="Answer"
        meta={
          answer.versionNo != null ? (
            <span className={styles.versionBadge}>
              Answer Version {answer.versionNo}
              {answer.supersedes != null && <span className={styles.supersedes}> · supersedes v{answer.supersedes}</span>}
            </span>
          ) : null
        }
      />
      <div className={styles.body}>
        {!hasContent && <p className={styles.empty}>The grounded answer will stream here once evidence is ready.</p>}
        {hasContent && (
          <p className={styles.text}>
            {segments.map((seg, i) =>
              seg.kind === "citation" ? (
                <button
                  key={i}
                  type="button"
                  className={styles.citation}
                  onClick={() => onCitationClick(seg.docId!, seg.section!)}
                  title="View supporting evidence"
                >
                  {seg.text}
                </button>
              ) : (
                <span key={i}>{seg.text}</span>
              ),
            )}
            {answer.isStreaming && <span className={styles.cursor} aria-hidden="true" />}
          </p>
        )}
      </div>
    </div>
  );
}
