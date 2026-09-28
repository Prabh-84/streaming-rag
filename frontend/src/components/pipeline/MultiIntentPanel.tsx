import { StatusDot } from "../common/StatusDot";
import type { SubqueryState } from "../../state/types";
import styles from "./MultiIntentPanel.module.css";

function retrievalDotTone(started: boolean, done: boolean): "idle" | "warn" | "live" {
  if (done) return "live";
  if (started) return "warn";
  return "idle";
}

export function MultiIntentPanel({ subqueries }: { subqueries: SubqueryState[] }) {
  if (subqueries.length === 0) return null;

  return (
    <div className={styles.wrap}>
      <p className={styles.heading}>
        Detected intent{subqueries.length > 1 ? "s" : ""} ({subqueries.length})
      </p>
      <ol className={styles.list}>
        {subqueries.map((sq, i) => (
          <li key={sq.subQueryId} className={styles.item}>
            <div className={styles.itemHead}>
              <span className={styles.index}>{i + 1}</span>
              <span className={styles.intentLabel}>{sq.intentLabel}</span>
              {sq.trigger && <span className={styles.trigger}>{sq.trigger}</span>}
            </div>
            <p className={styles.text}>&ldquo;{sq.text}&rdquo;</p>
            <div className={styles.stateRow}>
              <span className={styles.stateItem}>
                <StatusDot tone={retrievalDotTone(sq.retrievalStarted.dense, !!sq.retrievalCompleted.dense)} />
                dense
              </span>
              <span className={styles.stateItem}>
                <StatusDot tone={retrievalDotTone(sq.retrievalStarted.sparse, !!sq.retrievalCompleted.sparse)} />
                sparse
              </span>
              {sq.rerankedChunkIds.length > 0 && (
                <span className={styles.stateItem}>
                  <StatusDot tone="live" />
                  {sq.rerankedChunkIds.length} evidence kept
                </span>
              )}
              {sq.uncertainty && (
                <span className={styles.uncertain}>
                  <StatusDot tone="warn" />
                  uncertain: {sq.uncertainty.reason}
                </span>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
