"""Strongly typed data model for the streaming acoustic-expression system.

Everything here is a plain, framework-agnostic dataclass — no numpy, no
livekit, no torch — for the same reason `backchannel/events.py` stays plain:
these types are shared by the pipeline, the benchmark harness, and (via
`.to_dict()`) the JSON/event-log/UI boundary, and none of those should have
to pull in the ML stack just to read a timestamp.

Naming and terminology matter here per the assignment brief: these are
*acoustically expressed conversational signals* derived from prosody/audio,
never a claim about a person's true internal emotional state. Field names
(`frustration`, `uncertainty`, `energy`) are kept short for the wire format,
but every place they're displayed (UI, README) is expected to spell out the
"acoustic" qualifier — see apps/web/components/Timeline.tsx labels and the
README's "Terminology" section.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal

ModelKind = Literal["mock_dsp", "real_wav2vec2_er"]


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """Static description of an acoustic model, independent of any one
    prediction. Reported once at startup and attached to every prediction's
    metadata so a benchmark run or a debugging session can tell exactly
    which model (and version) produced a given number."""

    name: str
    version: str
    kind: ModelKind
    sample_rate: int
    min_duration_ms: float
    device: str  # "cpu" | "cuda"
    is_simulated: bool  # True for the DSP/mock model — never claim ML sophistication it doesn't have
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RawModelOutput:
    """What an `AcousticModel.predict()` call returns for one window.

    `energy` is deliberately absent here — it is a direct measurement of the
    waveform (RMS loudness), not something any model "predicts", and is
    computed once in the pipeline (see `acoustic/features.py`) regardless of
    which model is plugged in. Keeping it out of this type keeps that
    boundary honest: a model swap can never silently change how energy is
    computed.
    """

    frustration: float
    uncertainty: float
    confidence: float
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class WindowInfo:
    """The audio window a prediction was computed from."""

    start_ts: float  # clock() time of the first sample used — the "relevant speech" reference point
    end_ts: float  # clock() time the window closed / became ready for inference
    duration_ms: float
    sample_count: int
    is_partial: bool  # True if shorter than the configured WINDOW_SIZE_MS (early in a turn)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LatencyBreakdown:
    """The full latency accounting the assignment explicitly asks for —
    see README "Latency methodology". `effective_detection_latency_ms` is
    the only number that should ever be reported on its own; the other
    three explain what it's made of.
    """

    audio_wait_ms: float  # time spent accumulating enough audio (window duration)
    queue_wait_ms: float  # time between "window ready" and "inference actually started"
    inference_latency_ms: float  # model input ready -> prediction produced
    effective_detection_latency_ms: float  # relevant speech occurred -> usable prediction (sum of the above)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AcousticPrediction:
    """One fully-instrumented prediction: the unit that gets logged, smoothed,
    benchmarked, and (eventually) rendered on the timeline."""

    produced_at: float  # clock() time this prediction became available
    window: WindowInfo
    latency: LatencyBreakdown
    frustration: float  # raw (pre-smoothing) model output
    uncertainty: float
    energy: float
    confidence: float
    model: ModelInfo
    smoothed_frustration: float
    smoothed_uncertainty: float
    smoothed_energy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "producedAt": self.produced_at,
            "window": self.window.to_dict(),
            "latency": self.latency.to_dict(),
            "frustration": self.frustration,
            "uncertainty": self.uncertainty,
            "energy": self.energy,
            "confidence": self.confidence,
            "smoothedFrustration": self.smoothed_frustration,
            "smoothedUncertainty": self.smoothed_uncertainty,
            "smoothedEnergy": self.smoothed_energy,
            "modelVersion": self.model.version,
            "modelKind": self.model.kind,
            "isSimulated": self.model.is_simulated,
        }


@dataclasses.dataclass
class ExpressionState:
    """The current, smoothed acoustic state for a call — what a consumer
    (backchannel policy, UI "current state" panel) actually reads. Mutable
    and cheap: one instance lives per call, updated in place as predictions
    arrive. Never persisted; see README "Per-call state" for why this is
    safe to lose on a worker crash."""

    frustration: float = 0.0
    uncertainty: float = 0.0
    energy: float = 0.0
    confidence: float = 0.0
    updated_at: float = 0.0
    sample_count: int = 0
    model_version: str = ""
    stale: bool = False  # set by the pipeline when no fresh prediction has arrived recently

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TextBaselineScore:
    """Output of the independent lexicon-based text-sentiment baseline —
    see acoustic/text_baseline.py and README "Acoustic vs text baseline"."""

    positivity: float
    frustration_text: float
    uncertainty_text: float
    matched_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
