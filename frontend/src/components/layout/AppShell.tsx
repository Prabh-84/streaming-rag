import type { ReactNode } from "react";
import styles from "./AppShell.module.css";

export function AppShell({
  navbar,
  left,
  center,
  right,
}: {
  navbar: ReactNode;
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
}) {
  return (
    <div className={styles.shell}>
      {navbar}
      <main className={styles.workspace}>
        <section className={styles.left} aria-label="Live transcript">
          {left}
        </section>
        <section className={styles.center} aria-label="RAG pipeline and answer">
          {center}
        </section>
        <section className={styles.right} aria-label="Evidence and system activity">
          {right}
        </section>
      </main>
    </div>
  );
}
