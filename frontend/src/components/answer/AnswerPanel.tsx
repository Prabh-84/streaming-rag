import { PanelHeader } from "../common/PanelHeader";
import { splitAnswerIntoSegments } from "./citationParsing";
import type { AnswerState } from "../../state/types";
import styles from "./AnswerPanel.module.css";

function AnswerVersionCard({
  version,
  onCitationClick,
}: {
  version: AnswerState;
  onCitationClick: (docId: string, section: string) => void;
}) {
  const segments = splitAnswerIntoSegments(version.text);

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span className={styles.versionBadge}>
          Answer Version {version.versionNo}
          {version.supersedes != null && <span className={styles.supersedes}> · supersedes v{version.supersedes}</span>}
        </span>
      </div>
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
        {version.isStreaming && <span className={styles.cursor} aria-hidden="true" />}
      </p>
    </div>
  );
}

export function AnswerPanel({
  answer,
  answerHistory,
  onCitationClick,
}: {
  answer: AnswerState;
  answerHistory: AnswerState[];
  onCitationClick: (docId: string, section: string) => void;
}) {
  // The in-progress draft (from ANSWER_DELTA) is shown as a trailing live card until its
  // ANSWER_VERSION_CREATED lands in answerHistory, at which point it is dropped here to avoid
  // rendering the same version twice.
  const hasLiveDraft =
    answer.isStreaming &&
    answer.text.length > 0 &&
    !answerHistory.some((v) => v.versionNo === answer.versionNo);
  const cards = hasLiveDraft ? [...answerHistory, answer] : answerHistory;

  return (
    <div className={styles.panel}>
      <PanelHeader title="Answer" />
      <div className={styles.body}>
        {cards.length === 0 && (
          <p className={styles.empty}>The grounded answer will stream here once evidence is ready.</p>
        )}
        {cards.map((version) => (
          <AnswerVersionCard key={version.versionNo ?? "draft"} version={version} onCitationClick={onCitationClick} />
        ))}
      </div>
    </div>
  );
}
