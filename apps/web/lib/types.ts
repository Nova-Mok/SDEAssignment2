// Mirrors apps/agent/backchannel/events.py's schema and the sqlite tables
// instrumentation.py creates. One schema, read by both the benchmark writer
// (Python) and this dashboard (TypeScript). See README "Event model".

export type ConfigLabel = "baseline" | "backchannel";

export type EventType =
  | "USER_SPEECH_START"
  | "USER_SPEECH_END"
  | "STT_INTERIM"
  | "STT_FINAL"
  | "EOT_PROBABILITY_UPDATED"
  | "EOT_DETECTED"
  | "BACKCHANNEL_CANDIDATE"
  | "BACKCHANNEL_SELECTED"
  | "BACKCHANNEL_SUPPRESSED"
  | "BACKCHANNEL_CANCELLED"
  | "BACKCHANNEL_AUDIO_START"
  | "BACKCHANNEL_AUDIO_END"
  | "BAD_BACKCHANNEL"
  | "AGENT_STATE_CHANGED"
  | "LLM_START"
  | "LLM_TTFT"
  | "TTS_START"
  | "TTS_FIRST_AUDIO"
  | "AGENT_RESPONSE_START"
  | "AGENT_RESPONSE_END"
  | "SESSION_SHUTDOWN"
  | "ERROR"
  // Acoustic-expression side path (Assignment 2) — see
  // apps/agent/acoustic/pipeline.py and README "Observability".
  | "ACOUSTIC_DISABLED"
  | "ACOUSTIC_AUDIO_WINDOW_READY"
  | "ACOUSTIC_INFERENCE_STARTED"
  | "ACOUSTIC_INFERENCE_COMPLETED"
  | "ACOUSTIC_PREDICTION_PRODUCED"
  | "ACOUSTIC_PREDICTION_STALE"
  | "ACOUSTIC_QUEUE_OVERFLOW"
  | "ACOUSTIC_INFERENCE_FAILED"
  | "ACOUSTIC_MODEL_UNAVAILABLE"
  | "ACOUSTIC_EXPRESSION_UPDATED";

// One acoustic prediction's payload, as published in
// ACOUSTIC_PREDICTION_PRODUCED's metadata_json (mirrors
// acoustic.types.AcousticPrediction.to_dict()) and over the live LiveKit
// data-channel message (topic "acoustic", mirrors ExpressionState.to_dict()
// plus a sentAt wall-clock timestamp — see livekit_adapter.publish_expression_update).
export interface AcousticPredictionPayload {
  producedAt: number;
  window: { startTs: number; endTs: number; durationMs: number; sampleCount: number; isPartial: boolean };
  latency: { audioWaitMs: number; queueWaitMs: number; inferenceLatencyMs: number; effectiveDetectionLatencyMs: number };
  frustration: number;
  uncertainty: number;
  energy: number;
  confidence: number;
  smoothedFrustration: number;
  smoothedUncertainty: number;
  smoothedEnergy: number;
  modelVersion: string;
  modelKind: string;
  isSimulated: boolean;
}

export interface LiveExpressionMessage {
  frustration: number;
  uncertainty: number;
  energy: number;
  confidence: number;
  updated_at: number;
  sample_count: number;
  model_version: string;
  stale: boolean;
  sentAt: number;
}

export interface EventRow {
  id: number;
  run_id: string;
  scenario_id: string;
  config: ConfigLabel;
  event_type: EventType;
  source: string;
  timestamp: number;
  wall_time: number;
  turn_id: number;
  decision_id: number;
  metadata_json: string;
}

export interface RunRow {
  run_id: string;
  scenario_id: string;
  config: ConfigLabel;
  run_index: number;
  started_at: number;
  response_latency_ms: number | null;
  backchannel_latency_ms: number | null;
  llm_ttft_ms: number | null;
  tts_first_audio_ms: number | null;
  stt_finalization_ms: number | null;
  backchannel_count: number;
  cancelled_count: number;
  bad_backchannel_count: number;
  suppressed_count: number;
  overlap_ms: number;
  notes_json: string;
}

export interface ScenarioRow {
  scenario_id: string;
  name: string;
  description: string;
  expected_behaviour: string;
}

export interface MetricSummary {
  n: number;
  mean: number | null;
  p50: number | null;
  p95: number | null;
  min: number | null;
  max: number | null;
}
