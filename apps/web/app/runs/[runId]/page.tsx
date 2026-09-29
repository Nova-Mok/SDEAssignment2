import Link from "next/link";
import Timeline from "../../../components/Timeline";
import { findComparableRun, getEventsForRun, getRun, summarizeAcousticLatency } from "../../../lib/db";
import { fmtMs } from "../../../lib/format";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const run = getRun(runId);
  const events = getEventsForRun(runId);

  if (!run) {
    return (
      <div className="container">
        <div className="empty-state">Run {runId} not found.</div>
      </div>
    );
  }

  const otherConfig = run.config === "baseline" ? "backchannel" : "baseline";
  const comparable = findComparableRun(run.scenario_id, otherConfig, run.run_index);
  const acoustic = summarizeAcousticLatency(events);

  return (
    <div className="container">
      <p className="small">
        <Link href={`/scenarios/${run.scenario_id}`}>&larr; {run.scenario_id}</Link>
      </p>
      <h1>
        Run detail <span className={`badge ${run.config === "backchannel" ? "ok" : "warn"}`}>{run.config}</span>
      </h1>
      <p className="subtitle">
        {run.run_id}, run #{run.run_index}
        {comparable && (
          <>
            {" · "}
            <Link href={`/runs/${comparable.run_id}`}>compare with {otherConfig} run #{run.run_index}</Link>
          </>
        )}
      </p>

      <div className="panel" style={{ marginBottom: 20, overflowX: "auto" }}>
        <Timeline events={events} />
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Latency</h2>
          <table>
            <tbody>
              <tr><td>Response latency</td><td>{fmtMs(run.response_latency_ms)}</td></tr>
              <tr><td>LLM TTFT</td><td>{fmtMs(run.llm_ttft_ms)}</td></tr>
              <tr><td>TTS first audio</td><td>{fmtMs(run.tts_first_audio_ms)}</td></tr>
              <tr><td>Backchannel latency (avg)</td><td>{fmtMs(run.backchannel_latency_ms)}</td></tr>
              <tr><td>STT finalization (scripted)</td><td>{fmtMs(run.stt_finalization_ms)}</td></tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Behaviour</h2>
          <table>
            <tbody>
              <tr><td>Backchannels played</td><td>{run.backchannel_count}</td></tr>
              <tr><td>Cancelled</td><td>{run.cancelled_count}</td></tr>
              <tr><td>Suppressed candidates</td><td>{run.suppressed_count}</td></tr>
              <tr><td>Bad backchannels</td><td>{run.bad_backchannel_count}</td></tr>
              <tr><td>User/agent overlap</td><td>{fmtMs(run.overlap_ms)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {acoustic && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <h2 style={{ marginTop: 0 }}>Acoustic latency (this run)</h2>
          <p className="small" style={{ marginTop: -8 }}>
            Computed from this run&apos;s own {acoustic.predictionCount} ACOUSTIC_PREDICTION_PRODUCED events.
            &quot;Effective detection&quot; is the honest end-to-end number (relevant speech → usable prediction,
            including window-fill wait) — see README &quot;Latency methodology&quot; for why inference latency alone
            is not reported as the headline figure.
          </p>
          <table>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}></th>
                <th style={{ textAlign: "left" }}>P50</th>
                <th style={{ textAlign: "left" }}>P95</th>
                <th style={{ textAlign: "left" }}>mean</th>
                <th style={{ textAlign: "left" }}>min</th>
                <th style={{ textAlign: "left" }}>max</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ textAlign: "left" }}><strong>Effective detection latency</strong></td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.effectiveDetection.p50)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.effectiveDetection.p95)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.effectiveDetection.mean)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.effectiveDetection.min)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.effectiveDetection.max)}</td>
              </tr>
              <tr>
                <td style={{ textAlign: "left" }}>Audio-wait (window fill)</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.audioWait.p50)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.audioWait.p95)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.audioWait.mean)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.audioWait.min)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.audioWait.max)}</td>
              </tr>
              <tr>
                <td style={{ textAlign: "left" }}>Model inference</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.inference.p50)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.inference.p95)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.inference.mean)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.inference.min)}</td>
                <td style={{ textAlign: "left" }}>{fmtMs(acoustic.inference.max)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <h2>Raw event log</h2>
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>t (s)</th>
              <th style={{ textAlign: "left" }}>event</th>
              <th style={{ textAlign: "left" }}>source</th>
              <th style={{ textAlign: "left" }}>metadata</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e) => (
              <tr key={e.id}>
                <td style={{ textAlign: "left" }}>{(e.timestamp - events[0].timestamp).toFixed(3)}</td>
                <td style={{ textAlign: "left" }}>{e.event_type}</td>
                <td style={{ textAlign: "left" }}>{e.source}</td>
                <td style={{ textAlign: "left" }}><code>{e.metadata_json}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
