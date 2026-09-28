import { PanelHeader } from "../common/PanelHeader";
import { EvidenceCard } from "./EvidenceCard";
import type { EvidenceItem } from "../../state/types";
import styles from "./EvidencePanel.module.css";

export function EvidencePanel({
  evidence,
  highlightedChunkId,
}: {
  evidence: EvidenceItem[];
  highlightedChunkId: string | null;
}) {
  const sorted = [...evidence].sort((a, b) => b.rerankScore - a.rerankScore);

  return (
    <div className={styles.panel}>
      <PanelHeader title="Evidence" meta={evidence.length > 0 ? `${evidence.length} selected by reranker` : null} />
      <div className={styles.list}>
        {sorted.length === 0 && <p className={styles.empty}>Evidence selected by the reranker will appear here.</p>}
        {sorted.map((item) => (
          <EvidenceCard key={item.chunkId} item={item} highlighted={item.chunkId === highlightedChunkId} />
        ))}
      </div>
    </div>
  );
}
