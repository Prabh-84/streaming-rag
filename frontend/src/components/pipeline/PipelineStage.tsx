import type { ReactNode } from "react";
import { StatusDot } from "../common/StatusDot";
import type { StageStatus } from "../../state/types";
import styles from "./PipelineStage.module.css";

const STATUS_META: Record<StageStatus, { tone: "live" | "warn" | "error" | "idle" | "accent"; pulse: boolean; label: string }> = {
  idle: { tone: "idle", pulse: false, label: "Idle" },
  waiting: { tone: "warn", pulse: true, label: "Waiting" },
  active: { tone: "accent", pulse: true, label: "Active" },
  completed: { tone: "live", pulse: false, label: "Done" },
  failed: { tone: "error", pulse: false, label: "Failed" },
};

export function PipelineStage({
  label,
  status,
  detail,
  children,
}: {
  label: string;
  status: StageStatus;
  detail?: string | null;
  children?: ReactNode;
}) {
  const meta = STATUS_META[status];
  return (
    <div className={`${styles.stage} ${styles[status]}`}>
      <div className={styles.connector} aria-hidden="true" />
      <div className={styles.row}>
        <StatusDot tone={meta.tone} pulse={meta.pulse} />
        <span className={styles.label}>{label}</span>
        <span className={styles.statusText}>{meta.label}</span>
      </div>
      {detail && <p className={styles.detail}>{detail}</p>}
      {children}
    </div>
  );
}
