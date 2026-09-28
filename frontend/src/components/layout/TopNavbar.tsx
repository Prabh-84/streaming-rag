import { StatusDot } from "../common/StatusDot";
import type { ConnectionState } from "../../api/wsClient";
import styles from "./TopNavbar.module.css";

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting",
  connected: "Connected",
  reconnecting: "Reconnecting",
  disconnected: "Disconnected",
};

const CONNECTION_TONE: Record<ConnectionState, "live" | "warn" | "idle"> = {
  connecting: "warn",
  connected: "live",
  reconnecting: "warn",
  disconnected: "idle",
};

export function TopNavbar({
  corpusLabel,
  connection,
  hasSession,
}: {
  corpusLabel: string | null;
  connection: ConnectionState | null;
  hasSession: boolean;
}) {
  return (
    <header className={styles.bar}>
      <div className={styles.brand}>
        <div className={styles.titleRow}>
          <StatusDot tone="accent" pulse />
          <h1 className={styles.title}>Streaming Live RAG</h1>
        </div>
        <p className={styles.subtitle}>Real-time grounded retrieval over streaming context</p>
      </div>

      <div className={styles.statusGroup}>
        {corpusLabel && (
          <div className={styles.statusItem}>
            <span className={styles.statusLabel}>Corpus</span>
            <span className={styles.statusValue}>{corpusLabel}</span>
          </div>
        )}
        {hasSession && connection && (
          <div className={styles.statusItem}>
            <span className={styles.statusLabel}>Session</span>
            <span className={styles.statusValue}>
              <StatusDot tone={CONNECTION_TONE[connection]} pulse={connection === "connected"} />
              {CONNECTION_LABEL[connection]}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
