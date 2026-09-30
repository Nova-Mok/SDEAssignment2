"use client";

import { Room, RoomEvent, Track } from "livekit-client";
import { useRef, useState } from "react";
import type { LiveExpressionMessage } from "../../lib/types";

type Status = "idle" | "connecting" | "connected" | "disconnected" | "error";

const MAX_HISTORY = 60;

export default function LiveDemoPage() {
  const [mode, setMode] = useState<"baseline" | "backchannel">("backchannel");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [roomName, setRoomName] = useState<string>("");
  const [expression, setExpression] = useState<LiveExpressionMessage | null>(null);
  const [audioToUiLatencyMs, setAudioToUiLatencyMs] = useState<number | null>(null);
  const [history, setHistory] = useState<LiveExpressionMessage[]>([]);
  const roomRef = useRef<Room | null>(null);
  const audioContainerRef = useRef<HTMLDivElement | null>(null);

  async function connect() {
    setStatus("connecting");
    setError(null);
    try {
      const res = await fetch("/api/live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "failed to create room");

      const room = new Room();
      roomRef.current = room;

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const el = track.attach() as HTMLAudioElement;
          el.autoplay = true;
          audioContainerRef.current?.appendChild(el);
        }
      });
      room.on(RoomEvent.Disconnected, () => setStatus("disconnected"));

      // Acoustic expression state, pushed by the agent over a LiveKit data
      // message (topic "acoustic") every time a new smoothed prediction is
      // produced — see livekit_adapter.publish_expression_update(). No
      // polling: this is the real-time "while the user is talking" display
      // the assignment asks for (Phase 15), and doubles as our only
      // in-environment measurement of audio -> UI transport latency (see
      // README "Audio -> UI latency" for why the rest of that budget —
      // acoustic processing + detection latency — is measured separately
      // in benchmarks/acoustic_scenario_report.py rather than here).
      room.on(RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
        if (topic !== "acoustic") return;
        try {
          const msg = JSON.parse(new TextDecoder().decode(payload)) as LiveExpressionMessage;
          setExpression(msg);
          setAudioToUiLatencyMs((Date.now() / 1000 - msg.sentAt) * 1000);
          setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), msg]);
        } catch {
          // Malformed/unexpected payload on this topic — never let a display
          // bug affect the call; just drop this one update.
        }
      });

      await room.connect(data.url, data.token);
      await room.localParticipant.setMicrophoneEnabled(true);
      setRoomName(data.roomName);
      setStatus("connected");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }

  async function disconnect() {
    await roomRef.current?.disconnect();
    roomRef.current = null;
    setStatus("idle");
    setExpression(null);
    setAudioToUiLatencyMs(null);
    setHistory([]);
    if (audioContainerRef.current) audioContainerRef.current.innerHTML = "";
  }

  return (
    <div className="container">
      <h1>Live Demo</h1>
      <p className="subtitle">
        Talk to the real agent (real LiveKit room, real AWS STT/LLM/TTS). Toggle backchanneling and notice
        the difference, or don&apos;t. If it&apos;s working right, the real response shouldn&apos;t feel any slower either way.
      </p>

      <div className="panel">
        <div className="toggle-row" style={{ marginBottom: 16 }}>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "baseline"}
              onChange={() => setMode("baseline")}
              disabled={status === "connected" || status === "connecting"}
            />{" "}
            Backchannel OFF (baseline)
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "backchannel"}
              onChange={() => setMode("backchannel")}
              disabled={status === "connected" || status === "connecting"}
            />{" "}
            Backchannel ON
          </label>
        </div>

        {status === "connected" || status === "connecting" ? (
          <button className="btn secondary" onClick={disconnect} disabled={status === "connecting"}>
            Disconnect
          </button>
        ) : (
          <button className="btn" onClick={connect}>
            Connect &amp; enable mic
          </button>
        )}

        <div className="small" style={{ marginTop: 12 }}>
          Status: <strong>{status}</strong>
          {roomName && <> · room {roomName}</>}
          {error && <span style={{ color: "var(--bad)" }}> · {error}</span>}
        </div>
      </div>

      <div ref={audioContainerRef} style={{ display: "none" }} />

      {mode === "backchannel" && (status === "connected" || status === "connecting") && (
        <div className="panel" style={{ marginTop: 16 }}>
          <h2 style={{ marginTop: 0 }}>Acoustic state (live, while you speak)</h2>
          {expression ? (
            <>
              <div className="grid-3" style={{ display: "flex", gap: 16, marginBottom: 12 }}>
                <AcousticMeter label="Acoustic Frustration Signal" value={expression.frustration} color="#f87171" />
                <AcousticMeter label="Acoustic Uncertainty Signal" value={expression.uncertainty} color="#fbbf24" />
                <AcousticMeter label="Speaking Energy" value={expression.energy} color="#5b8cff" />
              </div>
              <div className="small">
                confidence {expression.confidence.toFixed(2)} · model {expression.model_version}
                {audioToUiLatencyMs !== null && <> · audio→UI ≈ {Math.max(0, audioToUiLatencyMs).toFixed(0)}ms</>}
              </div>
              <MiniSparkline history={history} />
            </>
          ) : (
            <div className="small">Waiting for the first prediction (needs ~0.5s of speech)…</div>
          )}
        </div>
      )}

      <h2>What to expect</h2>
      <div className="panel small">
        <p>
          <strong>Backchannel ON:</strong> while you&apos;re talking (especially a longer sentence), you should
          occasionally hear a short &quot;mm-hmm&quot;/&quot;okay&quot;/&quot;right&quot; layered under your own audio, on a
          track independent of the agent&apos;s real reply.
        </p>
        <p>
          <strong>Backchannel OFF:</strong> silence until you stop talking, then the agent responds. That&apos;s the
          classic behaviour the assignment describes as feeling unnatural on long turns.
        </p>
        <p>Requires browser microphone permission. Each connect creates a fresh room and disposes it on disconnect.</p>
        <p>
          <strong>Acoustic state:</strong> shown only for backchannel-mode connections. These are acoustically
          expressed conversational signals derived from your voice&apos;s delivery (energy, pitch stability, pauses),
          not a claim about your actual emotional state.
        </p>
      </div>
    </div>
  );
}

function AcousticMeter({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div style={{ flex: 1 }}>
      <div className="small" style={{ marginBottom: 4 }}>
        {label.toUpperCase()} <strong>{value.toFixed(2)}</strong>
      </div>
      <div style={{ background: "#232838", borderRadius: 4, height: 8, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color }} />
      </div>
    </div>
  );
}

function MiniSparkline({ history }: { history: { frustration: number; uncertainty: number; energy: number }[] }) {
  if (history.length < 2) return null;
  const width = 400;
  const height = 40;
  const barWidth = width / history.length;

  function series(pick: (h: (typeof history)[number]) => number, color: string) {
    return history.map((h, i) => {
      const v = Math.max(0, Math.min(1, pick(h)));
      const barHeight = Math.max(1, v * height);
      return <rect key={i} x={i * barWidth} y={height - barHeight} width={Math.max(1, barWidth - 1)} height={barHeight} fill={color} opacity={0.85} />;
    });
  }

  return (
    <div style={{ marginTop: 12 }}>
      <div className="small" style={{ marginBottom: 4 }}>
        Last {history.length} samples (frustration / uncertainty / energy)
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <svg width={width} height={height}>{series((h) => h.frustration, "#f87171")}</svg>
        <svg width={width} height={height}>{series((h) => h.uncertainty, "#fbbf24")}</svg>
        <svg width={width} height={height}>{series((h) => h.energy, "#5b8cff")}</svg>
      </div>
    </div>
  );
}
