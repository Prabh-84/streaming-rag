import { wsBaseUrl } from "./config";
import { WS_AUTH_FAILED, WS_UNKNOWN_SESSION } from "./types";
import type { TelemetryEvent, TranscriptChunkPayload } from "./types";

export type ConnectionState = "connecting" | "connected" | "reconnecting" | "disconnected";

export type ConnectionErrorReason = "auth_failed" | "unknown_session" | "network";

interface StreamClientOptions {
  /** The relative ws_url the backend returned from POST /session (includes the auth token). */
  wsUrlPath: string;
  onEvent: (event: TelemetryEvent) => void;
  onStateChange: (state: ConnectionState) => void;
  onError: (reason: ConnectionErrorReason) => void;
  /** Called once per successful (re)connection, after the socket is open. */
  onOpen?: (isReconnect: boolean) => void;
}

const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000];

/**
 * Thin wrapper around the real WS /session/{id}/stream protocol (app/api/stream.py). Reconnects
 * automatically on an unexpected drop by opening a new socket at the same ws_url - the backend's
 * own SESSION_RESYNC-on-reconnect behavior (REQ-STREAM-03) then does the actual state recovery;
 * this client does not simulate or invent that event.
 */
export class StreamClient {
  private socket: WebSocket | null = null;
  private closedByCaller = false;
  private hasConnectedOnce = false;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly options: StreamClientOptions) {}

  connect(): void {
    this.closedByCaller = false;
    this.open();
  }

  private open(): void {
    const isReconnect = this.hasConnectedOnce;
    this.options.onStateChange(isReconnect ? "reconnecting" : "connecting");

    const url = `${wsBaseUrl}${this.options.wsUrlPath}`;
    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempt = 0;
      this.hasConnectedOnce = true;
      this.options.onStateChange("connected");
      this.options.onOpen?.(isReconnect);
    };

    socket.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data as string) as TelemetryEvent;
        if (!event || typeof event.event_type !== "string") {
          throw new Error("malformed event envelope");
        }
        this.options.onEvent(event);
      } catch (err) {
        // A malformed frame must never crash the UI - log for developers, ignore for the judge.
        console.error("Received a malformed WebSocket event", err);
      }
    };

    socket.onclose = (closeEvent) => {
      if (this.closedByCaller) {
        this.options.onStateChange("disconnected");
        return;
      }
      if (closeEvent.code === WS_AUTH_FAILED) {
        this.options.onError("auth_failed");
        this.options.onStateChange("disconnected");
        return;
      }
      if (closeEvent.code === WS_UNKNOWN_SESSION) {
        this.options.onError("unknown_session");
        this.options.onStateChange("disconnected");
        return;
      }
      this.scheduleReconnect();
    };

    socket.onerror = () => {
      // The browser also fires onclose right after onerror for a failed connection; the actual
      // reconnect scheduling happens there so it isn't duplicated here.
    };
  }

  private scheduleReconnect(): void {
    const delay =
      RECONNECT_DELAYS_MS[Math.min(this.reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)];
    this.reconnectAttempt += 1;
    this.options.onStateChange("reconnecting");
    this.reconnectTimer = setTimeout(() => {
      if (!this.closedByCaller) this.open();
    }, delay);
  }

  sendTranscriptChunk(payload: TranscriptChunkPayload): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify({ event_type: "TRANSCRIPT_CHUNK", payload }));
    return true;
  }

  close(): void {
    this.closedByCaller = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.socket?.close();
    this.socket = null;
  }
}
