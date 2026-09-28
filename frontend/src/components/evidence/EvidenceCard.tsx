import { useState } from "react";
import type { EvidenceItem } from "../../state/types";
import styles from "./EvidenceCard.module.css";

export function EvidenceCard({ item, highlighted }: { item: EvidenceItem; highlighted: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const hasCitation = item.claimText != null;

  return (
    <article className={`${styles.card} ${highlighted ? styles.highlighted : ""}`} id={`evidence-${item.chunkId}`}>
      <button type="button" className={styles.summary} onClick={() => setExpanded((v) => !v)} aria-expanded={expanded}>
        <div className={styles.headline}>
          {item.docId ? (
            <span className={styles.doc}>
              {item.docId} <span className={styles.section}>{item.section}</span>
            </span>
          ) : (
            <span className={styles.docPending}>Retrieved evidence</span>
          )}
          <span className={styles.score}>{item.rerankScore.toFixed(2)}</span>
        </div>
        {item.claimText && <p className={styles.excerpt}>{item.claimText}</p>}
        {!hasCitation && <p className={styles.pending}>Not cited in the final answer</p>}
      </button>
      {expanded && (
        <dl className={styles.details}>
          <dt>chunk_id</dt>
          <dd>{item.chunkId}</dd>
          <dt>sub_query_id</dt>
          <dd>{item.subQueryId}</dd>
          <dt>rerank score</dt>
          <dd>{item.rerankScore}</dd>
        </dl>
      )}
    </article>
  );
}
