import type { ReactNode } from "react";
import styles from "./Banner.module.css";

export function Banner({
  tone,
  children,
  onDismiss,
}: {
  tone: "error" | "info" | "warn";
  children: ReactNode;
  onDismiss?: () => void;
}) {
  return (
    <div className={`${styles.banner} ${styles[tone]}`} role={tone === "error" ? "alert" : "status"}>
      <span className={styles.message}>{children}</span>
      {onDismiss && (
        <button type="button" className={styles.dismiss} onClick={onDismiss} aria-label="Dismiss">
          ×
        </button>
      )}
    </div>
  );
}
