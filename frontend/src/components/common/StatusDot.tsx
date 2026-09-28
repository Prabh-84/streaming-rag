import styles from "./StatusDot.module.css";

export type DotTone = "live" | "warn" | "error" | "idle" | "accent";

export function StatusDot({ tone, pulse = false }: { tone: DotTone; pulse?: boolean }) {
  return <span className={`${styles.dot} ${styles[tone]} ${pulse ? styles.pulse : ""}`} aria-hidden="true" />;
}
