import { useEffect, useRef, useState, type FormEvent } from "react";
import { DEMO_QUERY_LABEL, NORTHSTAR_DEMO_SCRIPT } from "../../demo/demoScript";
import styles from "./DemoControls.module.css";

type PlaybackState = "idle" | "playing" | "done";

export function DemoControls({
  onSendChunk,
  disabled,
}: {
  onSendChunk: (textDelta: string, isFinal: boolean, tOffsetMs: number) => boolean;
  disabled: boolean;
}) {
  const [playback, setPlayback] = useState<PlaybackState>("idle");
  const [manualText, setManualText] = useState("");
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const startDemo = () => {
    if (playback === "playing" || disabled) return;
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setPlayback("playing");

    let elapsed = 0;
    NORTHSTAR_DEMO_SCRIPT.forEach((chunk, i) => {
      elapsed += chunk.sendDelayMs;
      const timer = setTimeout(() => {
        onSendChunk(chunk.textDelta, chunk.isFinal, chunk.tOffsetMs);
        if (i === NORTHSTAR_DEMO_SCRIPT.length - 1) setPlayback("done");
      }, elapsed);
      timers.current.push(timer);
    });
  };

  const sendManual = (e: FormEvent) => {
    e.preventDefault();
    const text = manualText.trim();
    if (!text || disabled) return;
    onSendChunk(text, true, 0);
    setManualText("");
  };

  return (
    <div className={styles.bar}>
      <div className={styles.demo}>
        <button type="button" className={styles.startButton} onClick={startDemo} disabled={disabled || playback === "playing"}>
          {playback === "playing" ? "Sending transcript…" : "Start Demo"}
        </button>
        <span className={styles.demoLabel} title={DEMO_QUERY_LABEL}>
          {playback === "playing"
            ? "Streaming the demo transcript…"
            : playback === "done"
              ? "Demo transcript complete"
              : "Plays a real, timestamped transcript through the live WebSocket API"}
        </span>
      </div>

      <form className={styles.manual} onSubmit={sendManual}>
        <input
          type="text"
          value={manualText}
          onChange={(e) => setManualText(e.target.value)}
          placeholder="Or type a question and send it as a live transcript chunk…"
          disabled={disabled}
        />
        <button type="submit" className={styles.sendButton} disabled={disabled || manualText.trim().length === 0}>
          Send
        </button>
      </form>
    </div>
  );
}
