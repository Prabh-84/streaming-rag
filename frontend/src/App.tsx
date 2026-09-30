import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "./components/layout/AppShell";
import { TopNavbar } from "./components/layout/TopNavbar";
import { TranscriptPanel } from "./components/transcript/TranscriptPanel";
import { PipelineView } from "./components/pipeline/PipelineView";
import { AnswerPanel } from "./components/answer/AnswerPanel";
import { EvidencePanel } from "./components/evidence/EvidencePanel";
import { ActivityPanel } from "./components/activity/ActivityPanel";
import { ConnectPanel } from "./components/controls/ConnectPanel";
import { DemoControls } from "./components/controls/DemoControls";
import { Banner } from "./components/common/Banner";
import { setStoredApiKey } from "./api/config";
import { friendlyErrorMessage } from "./state/friendlyError";
import { useStreamingSession } from "./state/useStreamingSession";
import styles from "./App.module.css";

function humanizeCorpusId(corpusId: string): string {
  return corpusId
    .split(/[_-]+/)
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

export default function App() {
  const {
    state,
    phase,
    sessionId,
    corpusId,
    lifecycleError,
    start,
    sendChunk,
    dismissResyncNotice,
    dismissError,
  } = useStreamingSession();

  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);
  const highlightTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleConnect = useCallback(
    (apiKey: string, requestedCorpusId: string) => {
      setStoredApiKey(apiKey);
      void start(apiKey, requestedCorpusId);
    },
    [start],
  );

  const handleCitationClick = useCallback(
    (docId: string, section: string) => {
      const match = state.evidence.find((e) => e.docId === docId && e.section === section);
      if (!match) return;
      setHighlightedChunkId(match.chunkId);
      document.getElementById(`evidence-${match.chunkId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      if (highlightTimer.current) clearTimeout(highlightTimer.current);
      highlightTimer.current = setTimeout(() => setHighlightedChunkId(null), 3000);
    },
    [state.evidence],
  );

  const isReady = phase === "ready";

  useEffect(() => {
    if (state.lastError) console.error("Backend reported a processing error:", state.lastError);
  }, [state.lastError]);

  return (
    <>
      {!isReady && <ConnectPanel phase={phase} error={lifecycleError} onConnect={handleConnect} />}

      {sessionId && (
        <AppShell
          navbar={
            <TopNavbar
              corpusLabel={corpusId ? humanizeCorpusId(corpusId) : null}
              connection={state.connection}
              hasSession={Boolean(sessionId)}
            />
          }
          left={<TranscriptPanel chunks={state.transcript} controllerBusy={state.pipeline.stages.CONTROLLER !== "idle"} />}
          center={
            <div className={styles.centerColumn}>
              <DemoControls onSendChunk={sendChunk} disabled={!isReady} />
              {state.resyncNotice && (
                <Banner tone="info" onDismiss={dismissResyncNotice}>
                  Session resynchronized
                  {state.resyncNotice.latestAnswerVersion != null &&
                    ` · latest answer version ${state.resyncNotice.latestAnswerVersion}`}
                </Banner>
              )}
              {state.lastError && (
                <Banner tone="error" onDismiss={dismissError}>
                  {friendlyErrorMessage(state.lastError)}
                </Banner>
              )}
              <PipelineView state={state} />
              <AnswerPanel answer={state.answer} answerHistory={state.answerHistory} onCitationClick={handleCitationClick} />
            </div>
          }
          right={
            <div className={styles.rightColumn}>
              <EvidencePanel evidence={state.evidence} highlightedChunkId={highlightedChunkId} />
              <ActivityPanel items={state.activity} />
            </div>
          }
        />
      )}
    </>
  );
}
