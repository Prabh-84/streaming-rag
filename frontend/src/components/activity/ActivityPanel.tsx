import { useState } from "react";
import { PanelHeader } from "../common/PanelHeader";
import type { ActivityItem } from "../../state/types";
import styles from "./ActivityPanel.module.css";

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString(undefined, { hour12: false }) + `.${String(d.getMilliseconds()).padStart(3, "0")}`;
}

function ActivityRow({ item }: { item: ActivityItem }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <li className={styles.row}>
      <button
        type="button"
        className={styles.rowButton}
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <span className={styles.time}>{formatTime(item.timestamp)}</span>
        <span className={styles.label}>{item.label}</span>
      </button>
      {item.detail && !expanded && <p className={styles.detail}>{item.detail}</p>}
      {expanded && <pre className={styles.raw}>{JSON.stringify(item.raw, null, 2)}</pre>}
    </li>
  );
}

export function ActivityPanel({ items }: { items: ActivityItem[] }) {
  const [collapsed, setCollapsed] = useState(false);
  const ordered = [...items].reverse();

  return (
    <div className={styles.panel}>
      <PanelHeader
        title="Live Activity"
        meta={items.length > 0 ? `${items.length} event(s)` : null}
        action={
          <button
            type="button"
            className={styles.toggle}
            onClick={() => setCollapsed((v) => !v)}
            aria-expanded={!collapsed}
          >
            {collapsed ? "Show" : "Hide"}
          </button>
        }
      />
      {!collapsed && (
        <ul className={styles.list}>
          {ordered.length === 0 && <li className={styles.empty}>No activity yet.</li>}
          {ordered.map((item) => (
            <ActivityRow key={item.id} item={item} />
          ))}
        </ul>
      )}
    </div>
  );
}
