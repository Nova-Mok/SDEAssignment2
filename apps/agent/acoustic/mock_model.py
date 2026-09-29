"""The default runtime acoustic model: a deterministic, DSP/prosody-feature
model — no neural network, no weights to download, sub-millisecond on CPU.

Why this is the *default* rather than a fallback stub: `acoustic/real_model.py`
wraps a genuine pretrained speech model, and it was actually downloaded and
benchmarked on this machine (see README "Model selection" for the measured
numbers) — CPU inference for it lands around 100-170ms per window. Against
this system's INFERENCE_STRIDE_MS=250ms default, that leaves too little
headroom to stay real-time once you account for scheduling jitter and
concurrent calls on one host, so it isn't safe to run as the *default* live
model on CPU. This module exists per the assignment's explicit fallback
clause: "if the chosen model cannot practically run in the current
environment, implement a deterministic local model, keep the real adapter
separate, clearly mark simulated results." That is exactly what this is —
a real, explainable signal-processing pipeline over real audio (not random
numbers), deterministic given the same waveform, and measured at ~15-30ms
per 1.5s window on this machine (see benchmarks/acoustic_latency_bench.py)
— roughly 5-10x cheaper than the real model's stride budget, not
sub-millisecond, but comfortably inside the 250ms stride either way.

Every value this model reports is derived from four measured acoustic
features (see acoustic/features.py): RMS energy, pitch (F0) via
autocorrelation, voiced-fraction, spectral centroid, and pause ratio. The
mapping from features to frustration/uncertainty is a hand-specified,
documented heuristic — not a validated psychoacoustic model — which is why
its ModelInfo.is_simulated is True and why the README repeatedly says
"acoustically expressed" rather than implying ground truth.
"""
from __future__ import annotations

import numpy as np

from . import features
from .model import ModelUnavailableError  # re-exported for convenience
from .types import ModelInfo, RawModelOutput

__all__ = ["DeterministicProsodyModel", "ModelUnavailableError"]

_VERSION = "mock-dsp-v1"


class DeterministicProsodyModel:
    def __init__(self, sample_rate: int = 16000) -> None:
        self._info = ModelInfo(
            name="deterministic-prosody",
            version=_VERSION,
            kind="mock_dsp",
            sample_rate=sample_rate,
            min_duration_ms=300.0,
            device="cpu",
            is_simulated=True,
            description=(
                "Hand-specified heuristic over measured prosodic features "
                "(RMS energy, autocorrelation pitch, voiced ratio, spectral "
                "centroid, pause ratio). Deterministic given identical audio. "
                "Not a trained ML model — see README 'Model selection'."
            ),
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, audio: np.ndarray, sample_rate: int) -> RawModelOutput:
        if audio.size == 0:
            raise ValueError("cannot predict on empty audio window")

        energy = features.rms_energy(audio)
        f0, voiced_ratio = features.pitch_track(audio, sample_rate)
        voiced_f0 = f0[~np.isnan(f0)]
        if voiced_f0.size >= 2:
            pitch_mean = float(np.mean(voiced_f0))
            pitch_jitter = float(np.std(voiced_f0) / pitch_mean) if pitch_mean > 0 else 0.0
        else:
            pitch_jitter = 0.0
        pitch_jitter_norm = float(np.clip(pitch_jitter / 0.35, 0.0, 1.0))  # ~0.35 relative std ≈ very unstable pitch

        centroid_hz = features.spectral_centroid(audio, sample_rate)
        centroid_norm = float(np.clip(centroid_hz / 2500.0, 0.0, 1.0))  # 2.5kHz ≈ bright/tense upper end for speech

        pause_ratio = features.silence_ratio(audio, sample_rate)

        # --- Frustration proxy -------------------------------------------------
        # Raised energy + unstable pitch + a brighter/harder spectrum, damped
        # slightly by a strong voiced ratio (calm, fully-voiced, sustained
        # speech is not what frustration sounds like even when loud).
        frustration = (
            0.40 * energy
            + 0.30 * pitch_jitter_norm
            + 0.30 * centroid_norm
        )
        frustration *= 1.0 - 0.15 * voiced_ratio
        frustration = float(np.clip(frustration, 0.0, 1.0))

        # --- Uncertainty proxy ---------------------------------------------------
        # Hesitant delivery: pauses, unstable pitch, and quieter-than-average
        # speech (the inverse of confident, sustained energy).
        uncertainty = (
            0.45 * pause_ratio
            + 0.35 * pitch_jitter_norm
            + 0.20 * (1.0 - energy)
        )
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

        # --- Confidence ----------------------------------------------------------
        # Low when the window is mostly unvoiced/silent (nothing to measure)
        # or very short; high once there's a substantial voiced signal.
        duration_ms = audio.size / sample_rate * 1000
        completeness = float(np.clip(duration_ms / self._info.min_duration_ms, 0.0, 1.0))
        confidence = float(np.clip(0.25 + 0.55 * voiced_ratio * completeness, 0.05, 0.95))

        return RawModelOutput(
            frustration=frustration,
            uncertainty=uncertainty,
            confidence=confidence,
            extra={
                "pitchJitterNorm": pitch_jitter_norm,
                "spectralCentroidHz": centroid_hz,
                "voicedRatio": voiced_ratio,
                "pauseRatio": pause_ratio,
            },
        )
