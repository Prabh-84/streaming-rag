import type { ReactNode } from "react";
import styles from "./PanelHeader.module.css";

export function PanelHeader({ title, meta, action }: { title: string; meta?: ReactNode; action?: ReactNode }) {
  return (
    <div className={styles.header}>
      <div className={styles.left}>
        <h2 className={styles.title}>{title}</h2>
        {meta && <div className={styles.meta}>{meta}</div>}
      </div>
      {action}
    </div>
  );
}
