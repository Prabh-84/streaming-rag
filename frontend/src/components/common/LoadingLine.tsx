import styles from "./LoadingLine.module.css";

/** A small inline "working" indicator with real, caller-provided text - never a spinner alone. */
export function LoadingLine({ text }: { text: string }) {
  return (
    <div className={styles.row} role="status">
      <span className={styles.dots} aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>{text}</span>
    </div>
  );
}
