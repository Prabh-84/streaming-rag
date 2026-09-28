import { PanelHeader } from "../common/PanelHeader";
import type { TranscriptChunkState } from "../../state/types";
import styles from "./TranscriptPanel.module.css";

function statusLine(chunks: TranscriptChunkState[], controllerBusy: boolean): string {
  if (chunks.length === 0) return "Waiting for a transcript…";
  const last = chunks[chunks.length - 1];
  if (last.isFinal) return "Transcript complete";
  if (controllerBusy) return "Receiving transcript…";
  return "Listening for the next chunk…";
}

export function TranscriptPanel({
  chunks,
  controllerBusy,
}: {
  chunks: TranscriptChunkState[];
  controllerBusy: boolean;
}) {
  const isLatestActive = chunks.length > 0 && !chunks[chunks.length - 1].isFinal && controllerBusy;

  return (
    <div className={styles.panel}>
      <PanelHeader title="Live Transcript" meta={statusLine(chunks, controllerBusy)} />
      <div className={styles.list}>
        {chunks.length === 0 && (
          <p className={styles.empty}>Transcript chunks will appear here as they stream in.</p>
        )}
        {chunks.map((chunk, i) => {
          const isLatest = i === chunks.length - 1;
          return (
            <article
              key={chunk.seq}
              className={`${styles.chunk} ${isLatest && isLatestActive ? styles.active : ""} ${chunk.isFinal ? styles.final : ""}`}
            >
              <div className={styles.chunkMeta}>
                <span className={styles.seq}>#{chunk.seq}</span>
                <span className={styles.offset}>t+{chunk.tOffsetMs}ms</span>
                {chunk.isFinal && <span className={styles.finalBadge}>FINAL</span>}
              </div>
              <p className={styles.text}>{chunk.textDelta}</p>
            </article>
          );
        })}
      </div>
    </div>
  );
}
