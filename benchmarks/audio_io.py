"""Shared WAV loading for the acoustic benchmark scripts. All fixtures in
benchmarks/audio_fixtures/ are 16kHz mono 16-bit PCM (see
scripts/generate_acoustic_fixtures.py); this module is the one place that
assumption is encoded.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent / "audio_fixtures"


def load_wav_float32(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sample_rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def load_fixture(name: str) -> tuple[np.ndarray, int]:
    """`name` without the .wav extension, e.g. 'same_words_frustrated'."""
    return load_wav_float32(FIXTURES_DIR / f"{name}.wav")


def with_added_noise(audio: np.ndarray, snr_db: float = 8.0, seed: int = 0) -> np.ndarray:
    """Adds synthetic white noise at the given signal-to-noise ratio on top
    of REAL Polly speech — used for the "background noise" benchmark
    scenario. Labeled here (and everywhere it's used) as synthetically
    added noise on top of real synthesized speech, not a real noisy
    recording."""
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.square(audio))) or 1e-8
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)
