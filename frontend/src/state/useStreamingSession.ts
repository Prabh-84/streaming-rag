import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApiError, createSession } from "../api/restClient";
import type { TranscriptChunkPayload } from "../api/types";
import { ConnectionErrorReason, StreamClient } from "../api/wsClient";
import { sessionReducer } from "./sessionReducer";
import { createInitialSessionState } from "./types";

export type LifecyclePhase =
  | "idle"
  | "creating_session"
  | "connecting"
  | "ready"
  | "error";

export interface LifecycleError {
  message: string;
  cause: "session_creation" | ConnectionErrorReason;
}

export function useStreamingSession() {
  const [state, dispatch] = useReducer(sessionReducer, undefined, createInitialSessionState);
  const [phase, setPhase] = useState<LifecyclePhase>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [corpusId, setCorpusId] = useState<string | null>(null);
  const [lifecycleError, setLifecycleError] = useState<LifecycleError | null>(null);
  const clientRef = useRef<StreamClient | null>(null);
  const nextSeqRef = useRef(0);

  const teardown = useCallback(() => {
    clientRef.current?.close();
    clientRef.current = null;
  }, []);

  useEffect(() => teardown, [teardown]);

  const start = useCallback(
    async (apiKey: string, requestedCorpusId: string) => {
      teardown();
      setLifecycleError(null);
      setPhase("creating_session");
      nextSeqRef.current = 0;

      let response;
      try {
        response = await createSession(apiKey, { corpus_id: requestedCorpusId });
      } catch (err) {
        const message = err instanceof ApiError ? err.message : "Could not create a session.";
        setLifecycleError({ message, cause: "session_creation" });
        setPhase("error");
        return;
      }

      setSessionId(response.session_id);
      setCorpusId(requestedCorpusId);
      setPhase("connecting");

      const client = new StreamClient({
        wsUrlPath: response.ws_url,
        onEvent: (event) => dispatch({ type: "EVENT", event }),
        onStateChange: (connState) => {
          dispatch({ type: "CONNECTION_STATE", state: connState });
          if (connState === "connected") setPhase("ready");
        },
        onError: (reason) => {
          const message =
            reason === "auth_failed"
              ? "The API key was rejected by the backend."
              : reason === "unknown_session"
                ? "This session no longer exists (it may have expired)."
                : "Lost connection to the backend.";
          setLifecycleError({ message, cause: reason });
          setPhase("error");
        },
      });
      clientRef.current = client;
      client.connect();
    },
    [teardown],
  );

  const sendChunk = useCallback((textDelta: string, isFinal: boolean, tOffsetMs: number) => {
    const payload: TranscriptChunkPayload = {
      seq: nextSeqRef.current,
      text_delta: textDelta,
      t_offset_ms: tOffsetMs,
      is_final: isFinal,
    };
    nextSeqRef.current += 1;
    const sent = clientRef.current?.sendTranscriptChunk(payload) ?? false;
    if (sent) {
      dispatch({
        type: "CHUNK_SENT",
        chunk: { seq: payload.seq, textDelta, tOffsetMs, isFinal, sentAt: Date.now() },
      });
    }
    return sent;
  }, []);

  const disconnect = useCallback(() => {
    teardown();
    setPhase("idle");
    setSessionId(null);
    setCorpusId(null);
  }, [teardown]);

  const dismissResyncNotice = useCallback(() => dispatch({ type: "DISMISS_RESYNC_NOTICE" }), []);
  const dismissError = useCallback(() => dispatch({ type: "DISMISS_ERROR" }), []);

  return {
    state,
    phase,
    sessionId,
    corpusId,
    lifecycleError,
    start,
    sendChunk,
    disconnect,
    dismissResyncNotice,
    dismissError,
  };
}
