"""Phases 3/4/5/6/13 in one script: the 8 required benchmark scenarios, run
through the real streaming pipeline (default DeterministicProsodyModel),
reporting:

  - per-scenario acoustic signal trajectories (raw vs. smoothed)
  - true detection latency P50/P95, decomposed per the assignment's
    "audio_wait + queue_wait + inference = effective detection latency"
    breakdown (never reporting inference latency alone as "the" latency)
  - a concrete before/after smoothing example

Every fixture is real Amazon Polly neural-TTS speech (background_noise adds
synthetic white noise on top of one of them — see audio_io.with_added_noise)
— see scripts/generate_acoustic_fixtures.py. This script needs no AWS
credentials or network access; it only reads the committed WAV files, so
CI/reviewers can reproduce these exact numbers offline.

Usage: python -m benchmarks.acoustic_scenario_report [--repeats N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "apps" / "agent", REPO_ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from acoustic.config import AcousticConfig  # noqa: E402
from acoustic.mock_model import DeterministicProsodyModel  # noqa: E402
from acoustic.pipeline import AcousticStreamProcessor  # noqa: E402

from audio_io import load_fixture, with_added_noise  # noqa: E402

CONFIG = AcousticConfig(WINDOW_SIZE_MS=1500.0, INFERENCE_STRIDE_MS=250.0, MIN_AUDIO_REQUIRED_MS=500.0)


def _noisy_increasing_frustration():
    audio, sr = load_fixture("increasing_frustration")
    return with_added_noise(audio, snr_db=8.0), sr


SCENARIOS = {
    "same_words_different_delivery": lambda: load_fixture("same_words_frustrated"),
    "increasing_frustration": lambda: load_fixture("increasing_frustration"),
    "hesitant_speaker": lambda: load_fixture("hesitant_speaker"),
    "high_energy_speaker": lambda: load_fixture("high_energy_speaker"),
    "flat_delivery": lambda: load_fixture("same_words_flat"),
    "background_noise": _noisy_increasing_frustration,
    "long_conversation": lambda: load_fixture("long_conversation"),
    "hindi_hinglish": lambda: load_fixture("hindi_hinglish"),
}


async def _replay(audio: np.ndarray, sample_rate: int):
    model = DeterministicProsodyModel()
    proc = AcousticStreamProcessor(config=CONFIG, model=model)
    await proc.start()
    frame_len = int(sample_rate * 0.02)
    pos = 0
    t0 = time.monotonic()
    while pos + frame_len <= audio.size:
        proc.push_frame(time.monotonic(), audio[pos : pos + frame_len].astype(np.float32))
        pos += frame_len
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)  # let the last scheduled inference land
    history = proc.history
    await proc.stop()
    return history, t0


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


async def main(repeats: int) -> None:
    all_audio_wait: list[float] = []
    all_queue_wait: list[float] = []
    all_infer: list[float] = []
    all_detect: list[float] = []
    report: dict = {"scenarios": {}, "repeatsPerScenario": repeats}

    for name, loader in SCENARIOS.items():
        audio, sr = loader()
        last_run = None
        last_t0 = None
        for _ in range(repeats):
            history, t0 = await _replay(audio, sr)
            last_run, last_t0 = history, t0
            for p in history:
                all_audio_wait.append(p.latency.audio_wait_ms)
                all_queue_wait.append(p.latency.queue_wait_ms)
                all_infer.append(p.latency.inference_latency_ms)
                all_detect.append(p.latency.effective_detection_latency_ms)

        report["scenarios"][name] = {
            "durationS": round(audio.size / sr, 2),
            "numPredictions": len(last_run),
            "trajectory": [
                {
                    "tRelMs": round((p.produced_at - last_t0) * 1000, 1),
                    "frustration": round(p.frustration, 3),
                    "smoothedFrustration": round(p.smoothed_frustration, 3),
                    "uncertainty": round(p.uncertainty, 3),
                    "smoothedUncertainty": round(p.smoothed_uncertainty, 3),
                    "energy": round(p.energy, 3),
                    "smoothedEnergy": round(p.smoothed_energy, 3),
                    "confidence": round(p.confidence, 3),
                }
                for p in last_run
            ],
        }
        print(f"{name}: {len(last_run)} predictions over {audio.size / sr:.1f}s (x{repeats} repeats)")

    report["latency_ms"] = {
        "audio_wait": {"p50": _percentile(all_audio_wait, 0.5), "p95": _percentile(all_audio_wait, 0.95)},
        "queue_wait": {"p50": _percentile(all_queue_wait, 0.5), "p95": _percentile(all_queue_wait, 0.95)},
        "inference": {"p50": _percentile(all_infer, 0.5), "p95": _percentile(all_infer, 0.95)},
        "effective_detection": {"p50": _percentile(all_detect, 0.5), "p95": _percentile(all_detect, 0.95)},
        "n": len(all_detect),
    }

    print(f"\n=== True latency (n={len(all_detect)} predictions across all scenarios x{repeats} repeats) ===")
    for key in ("audio_wait", "queue_wait", "inference", "effective_detection"):
        s = report["latency_ms"][key]
        print(f"  {key:<22} p50={s['p50']:.1f}ms  p95={s['p95']:.1f}ms")
    print(
        "\nNote: 'inference' alone (model input ready -> prediction produced) is NOT the "
        "number to report as detection latency — 'effective_detection' (relevant speech -> "
        "usable prediction, including the WINDOW_SIZE_MS audio-fill wait) is the honest one."
    )

    # Phase 6: a concrete smoothing example, using increasing_frustration
    # (deliberately designed to have a real rising trend to smooth).
    inc = report["scenarios"]["increasing_frustration"]["trajectory"]
    print("\n=== Smoothing example (increasing_frustration, raw vs. smoothed frustration) ===")
    print(f"{'t(ms)':>8} {'raw':>8} {'smoothed':>10}")
    for p in inc:
        print(f"{p['tRelMs']:>8.0f} {p['frustration']:>8.3f} {p['smoothedFrustration']:>10.3f}")

    out_path = REPO_ROOT / "data" / "acoustic_scenario_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.repeats))
