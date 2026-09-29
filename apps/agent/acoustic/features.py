"""Low-level acoustic feature extraction shared by every model
implementation. Kept separate from any one model so that "energy" in
particular is computed identically regardless of which model (mock or real)
is active — it is a direct measurement of the waveform, not a model
prediction, and a model swap must never silently change its meaning.
"""
from __future__ import annotations

import numpy as np

# RMS of "typically loud speech" at 16-bit PCM full scale, chosen empirically
# from the repo's own Polly-generated fixtures (see
# benchmarks/audio_fixtures/manifest.json) — not a universal acoustic
# constant, just a normalization reference so `energy` lands in a sane 0..1
# range for this system's inputs instead of clipping at ~1.0 for every voice.
_REFERENCE_RMS = 0.09


def rms_energy(audio: np.ndarray, reference_rms: float = _REFERENCE_RMS) -> float:
    """Normalized RMS loudness in [0, 1]. This IS "speaking energy" — no
    model involved, just `sqrt(mean(x^2))` against a fixed reference."""
    if audio.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return float(np.clip(rms / reference_rms, 0.0, 1.0))


def frame_signal(audio: np.ndarray, sample_rate: int, frame_ms: float = 30.0, hop_ms: float = 10.0) -> np.ndarray:
    """Split into overlapping sub-frames for pitch/voicing analysis.
    Returns shape (n_frames, frame_len); drops a trailing partial frame."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    n = audio.size
    if n < frame_len:
        return audio.reshape(1, -1) if n > 0 else np.zeros((0, frame_len), dtype=audio.dtype)
    n_frames = 1 + (n - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return audio[idx]


def _autocorrelation_pitch(frame: np.ndarray, sample_rate: int, fmin: float = 80.0, fmax: float = 400.0) -> float | None:
    """Time-domain autocorrelation pitch estimate for one sub-frame. Returns
    None for an unvoiced/silent frame (this is deliberate — voiced-fraction
    is itself a feature, computed by the caller from how many frames return
    a value here)."""
    windowed = frame * np.hanning(frame.size)
    if np.max(np.abs(windowed)) < 1e-4:
        return None
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[corr.size // 2 :]
    lag_min = int(sample_rate / fmax)
    lag_max = min(int(sample_rate / fmin), corr.size - 1)
    if lag_max <= lag_min:
        return None
    segment = corr[lag_min:lag_max]
    if segment.size == 0 or corr[0] <= 0:
        return None
    peak_lag = lag_min + int(np.argmax(segment))
    peak_val = corr[peak_lag]
    # Voicing threshold: a real periodic signal has a strong self-similarity
    # peak relative to zero-lag energy; noise/silence does not.
    if peak_val / corr[0] < 0.3:
        return None
    return sample_rate / peak_lag


def pitch_track(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """Returns (f0_per_frame in Hz with NaN for unvoiced, voiced_ratio)."""
    frames = frame_signal(audio, sample_rate)
    if frames.shape[0] == 0:
        return np.array([]), 0.0
    f0 = np.full(frames.shape[0], np.nan)
    for i, frame in enumerate(frames):
        pitch = _autocorrelation_pitch(frame, sample_rate)
        if pitch is not None:
            f0[i] = pitch
    voiced_ratio = float(np.mean(~np.isnan(f0))) if f0.size else 0.0
    return f0, voiced_ratio


def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    """Average spectral centroid in Hz, a brightness/tension proxy — a
    voice with more high-frequency energy relative to low sounds "harder" or
    "sharper", which is a real (if crude) acoustic correlate of vocal tension."""
    frames = frame_signal(audio, sample_rate, frame_ms=30.0, hop_ms=15.0)
    if frames.shape[0] == 0:
        return 0.0
    window = np.hanning(frames.shape[1])
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    total = np.sum(spectrum, axis=1)
    valid = total > 1e-6
    if not np.any(valid):
        return 0.0
    centroids = np.sum(spectrum[valid] * freqs[None, :], axis=1) / total[valid]
    return float(np.mean(centroids))


def silence_ratio(audio: np.ndarray, sample_rate: int, threshold_rms: float = 0.02, frame_ms: float = 30.0) -> float:
    """Fraction of sub-frames below an energy threshold — a pause/hesitation
    proxy, distinct from overall `energy` (a turn can be loud on average but
    still full of pauses)."""
    frames = frame_signal(audio, sample_rate, frame_ms=frame_ms, hop_ms=frame_ms)
    if frames.shape[0] == 0:
        return 0.0
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return float(np.mean(frame_rms < threshold_rms))
