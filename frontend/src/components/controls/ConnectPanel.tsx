import { useState, type FormEvent } from "react";
import { LoadingLine } from "../common/LoadingLine";
import { Banner } from "../common/Banner";
import { defaultCorpusId, getStoredApiKey } from "../../api/config";
import type { LifecyclePhase, LifecycleError } from "../../state/useStreamingSession";
import styles from "./ConnectPanel.module.css";

const PHASE_LABEL: Record<LifecyclePhase, string> = {
  idle: "",
  creating_session: "Creating session…",
  connecting: "Connecting…",
  ready: "",
  error: "",
};

export function ConnectPanel({
  phase,
  error,
  onConnect,
}: {
  phase: LifecyclePhase;
  error: LifecycleError | null;
  onConnect: (apiKey: string, corpusId: string) => void;
}) {
  const [apiKey, setApiKey] = useState(getStoredApiKey());
  const [corpusId, setCorpusId] = useState(defaultCorpusId);
  const busy = phase === "creating_session" || phase === "connecting";

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!apiKey.trim() || !corpusId.trim() || busy) return;
    onConnect(apiKey.trim(), corpusId.trim());
  };

  return (
    <div className={styles.overlay}>
      <form className={styles.card} onSubmit={submit}>
        <h2 className={styles.title}>Connect to the backend</h2>
        <p className={styles.hint}>
          Enter the API key for this deployment and the corpus to work against. The key is kept only
          in this tab's session storage - it is never written to disk or committed.
        </p>

        {error && <Banner tone="error">{error.message}</Banner>}

        <label className={styles.field}>
          <span>API key</span>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Bearer token"
            autoComplete="off"
            disabled={busy}
            required
          />
        </label>

        <label className={styles.field}>
          <span>Corpus ID</span>
          <input
            type="text"
            value={corpusId}
            onChange={(e) => setCorpusId(e.target.value)}
            placeholder={defaultCorpusId}
            disabled={busy}
            required
          />
        </label>

        <button type="submit" className={styles.submit} disabled={busy}>
          {busy ? "Connecting…" : "Connect"}
        </button>

        {busy && <LoadingLine text={PHASE_LABEL[phase]} />}
      </form>
    </div>
  );
}
