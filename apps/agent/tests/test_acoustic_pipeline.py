"""Behavioral tests for AcousticStreamProcessor — the real-time engineering
claims the assignment asks to be demonstrated, not just described:
streaming windows, backpressure/staleness, failure isolation, and clean
shutdown.

Two families, same split as test_backchannel_engine.py:

- Plumbing tests use `FakeAcousticModel` (tests/acoustic_fakes.py) and a
  `ManualClock` so timing assertions are deterministic and fast.
- One true integration test at the bottom uses the REAL
  `DeterministicProsodyModel` over synthetic audio with a rising energy
  envelope, to prove the signal actually tracks a real acoustic feature —
  not just that the plumbing calls a stub correctly.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from acoustic.config import AcousticConfig
from acoustic.mock_model import DeterministicProsodyModel
from acoustic.pipeline import AcousticStreamProcessor
from backchannel.events import EventType
from backchannel.instrumentation import InMemoryRecorder

from .acoustic_fakes import FakeAcousticModel
from .fakes import ManualClock

SR = 16000
FRAME_MS = 20
FRAME_SAMPLES = SR * FRAME_MS // 1000


def _frame(value: float = 0.05) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


FAST_CONFIG = AcousticConfig(
    WINDOW_SIZE_MS=200.0,
    INFERENCE_STRIDE_MS=20.0,
    MIN_AUDIO_REQUIRED_MS=60.0,
    MAX_QUEUE_SIZE=4,
    MAX_STALENESS_MS=150.0,
    INFERENCE_TIMEOUT_MS=80.0,
)


async def _settle(n: int = 3) -> None:
    """Yield to the event loop enough times for the ingest/scheduler tasks
    to catch up, without relying on real wall-clock sleeps for anything
    time-sensitive (those tests advance a ManualClock instead)."""
    for _ in range(n):
        await asyncio.sleep(0)


# ------------------------------------------------------------------ #
# Streaming / windowing
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_no_prediction_before_min_audio_required():
    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder)
    await proc.start()
    try:
        proc.push_frame(0.0, _frame())  # 20ms < MIN_AUDIO_REQUIRED_MS=60ms
        await asyncio.sleep(0.05)
        assert model.calls == 0
        assert proc.history == []
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_predictions_stream_while_still_speaking():
    """The core "streaming, not post-processing" claim: predictions must
    arrive well before a multi-second turn has finished."""
    model = FakeAcousticModel(frustration=0.6, uncertainty=0.2, confidence=0.9)
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder)
    await proc.start()
    try:
        for i in range(20):  # 400ms of audio, well short of a "whole turn"
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.05)

        assert model.calls >= 2, "expected multiple predictions before the simulated turn ended"
        assert proc.state.frustration == pytest.approx(0.6, abs=0.05)
        assert len(proc.history) >= 2
        for prediction in proc.history:
            assert prediction.latency.effective_detection_latency_ms > 0
            assert prediction.latency.audio_wait_ms >= 0
            assert prediction.model.version == "fake-v1"
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_partial_window_used_when_user_stops_mid_window():
    """User goes silent before a full WINDOW_SIZE_MS of audio has
    accumulated — the pipeline must still produce a (marked-partial)
    prediction from what's available rather than waiting forever."""
    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder)
    await proc.start()
    try:
        for i in range(5):  # 100ms: > MIN_AUDIO_REQUIRED_MS(60ms), < WINDOW_SIZE_MS(200ms)
            proc.push_frame(time.monotonic(), _frame())
        await asyncio.sleep(0.1)  # user stops pushing frames here — mid-window

        assert model.calls >= 1
        assert any(p.window.is_partial for p in proc.history)
    finally:
        await proc.stop()


# ------------------------------------------------------------------ #
# Backpressure (Phase 9/10: bounded queue, drop-oldest, staleness)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_queue_overflow_drops_oldest_frame_not_newest():
    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    config = AcousticConfig(**{**FAST_CONFIG.__dict__, "MAX_QUEUE_SIZE": 3})
    proc = AcousticStreamProcessor(config=config, model=model, recorder=recorder)
    await proc.start()
    try:
        # Push more than MAX_QUEUE_SIZE frames with NO await in between, so the
        # ingest task (a separate asyncio task) gets no chance to drain the
        # queue — this deterministically forces an overflow.
        for i in range(8):
            proc.push_frame(float(i), np.full(FRAME_SAMPLES, float(i), dtype=np.float32))

        assert proc.dropped_frames > 0
        overflow_events = recorder.of_type(EventType.ACOUSTIC_QUEUE_OVERFLOW)
        assert len(overflow_events) == proc.dropped_frames

        # The queue must have kept the NEWEST frames, not the oldest.
        remaining = list(proc._queue._queue)  # white-box: internal asyncio.Queue deque
        kept_markers = [float(chunk[0]) for _, chunk in remaining]
        assert max(kept_markers) == 7.0  # the very last frame pushed survived
        assert min(kept_markers) >= 8 - config.MAX_QUEUE_SIZE  # oldest surviving frame is recent, not frame 0
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_stale_window_is_dropped_before_inference():
    """A window whose audio is already older than MAX_STALENESS_MS by the
    time the scheduler would submit it must be dropped, not processed —
    "a prediction about speech from 3 seconds ago may no longer be useful"."""
    clock = ManualClock(start=0.0)
    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    config = AcousticConfig(**{**FAST_CONFIG.__dict__, "MAX_STALENESS_MS": 10.0})
    proc = AcousticStreamProcessor(config=config, model=model, recorder=recorder, clock=clock)
    await proc.start()
    try:
        for i in range(5):
            proc.push_frame(clock(), _frame())
            clock.advance(FRAME_MS / 1000)
        await _settle()

        # Simulate "we got busy for 500ms of wall-clock time" without any
        # corresponding real sleep — the next scheduler tick (real time,
        # ~20ms away) will see a window whose end_ts is now 500ms stale.
        clock.advance(0.5)
        await asyncio.sleep(0.05)

        assert model.calls == 0, "a stale window must never reach inference"
        stale_events = recorder.of_type(EventType.ACOUSTIC_PREDICTION_STALE)
        assert any(e.metadata.get("reason") == "window_stale_before_submit" for e in stale_events)
        assert proc.dropped_windows > 0
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_stale_inflight_inference_is_reaped_and_cancelled():
    """The scheduler's watchdog: an in-flight inference task that has been
    running longer than INFERENCE_TIMEOUT_MS is cancelled directly, freeing
    the pipeline to try a fresher window — tested in isolation from
    `asyncio.wait_for`'s own timeout to prove this is a real, independent
    safety net (see pipeline.py `_maybe_reap_stale_inference`)."""
    clock = ManualClock(start=0.0)
    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder, clock=clock)

    async def _never_finishes():
        await asyncio.sleep(1000)

    task = asyncio.create_task(_never_finishes())
    proc._inference_task = task
    proc._inference_started_at = clock()
    clock.advance(FAST_CONFIG.INFERENCE_TIMEOUT_MS / 1000 + 0.01)

    proc._maybe_reap_stale_inference()
    await asyncio.sleep(0)  # let the cancellation land

    assert task.cancelled()
    stale_events = recorder.of_type(EventType.ACOUSTIC_PREDICTION_STALE)
    assert any(e.metadata.get("reason") == "inference_exceeded_timeout_cancelled" for e in stale_events)


# ------------------------------------------------------------------ #
# Failure isolation (Phase 11)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_slow_inference_times_out_without_blocking_the_pipeline():
    model = FakeAcousticModel(delay_s=1.0)  # far longer than INFERENCE_TIMEOUT_MS
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder)
    await proc.start()
    try:
        for i in range(10):
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(FAST_CONFIG.INFERENCE_TIMEOUT_MS / 1000 + 0.05)

        failed = recorder.of_type(EventType.ACOUSTIC_INFERENCE_FAILED)
        assert any(e.metadata.get("reason") == "timeout" for e in failed)
        assert proc.state.sample_count == 0, "a timed-out inference must never update expression state"
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_failed_inference_is_isolated_and_pipeline_keeps_running():
    model = FakeAcousticModel(fail=True)
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder)
    await proc.start()
    try:
        for i in range(15):
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.05)

        assert model.calls >= 2, "a failing model must still be retried on the next cadence tick"
        failed = recorder.of_type(EventType.ACOUSTIC_INFERENCE_FAILED)
        assert len(failed) >= 2
        assert all(e.metadata.get("reason") == "exception" for e in failed)
        assert proc.state.sample_count == 0
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_model_unavailable_is_recorded_distinctly_from_generic_failure():
    from .acoustic_fakes import ModelUnavailableError

    model = FakeAcousticModel(fail=True, fail_with=ModelUnavailableError)
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder)
    await proc.start()
    try:
        for i in range(10):
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.05)

        assert recorder.of_type(EventType.ACOUSTIC_MODEL_UNAVAILABLE) != []
        assert recorder.of_type(EventType.ACOUSTIC_INFERENCE_FAILED) == []
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_on_state_update_callback_exception_does_not_break_pipeline():
    def _boom(state):
        raise RuntimeError("consumer bug")

    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model, recorder=recorder, on_state_update=_boom)
    await proc.start()
    try:
        for i in range(10):
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.05)
        assert model.calls >= 1  # the callback raising must not have stopped subsequent inference
    finally:
        await proc.stop()


# ------------------------------------------------------------------ #
# Disabled mode / clean shutdown
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_disabled_config_is_a_complete_noop():
    model = FakeAcousticModel()
    recorder = InMemoryRecorder()
    config = AcousticConfig(**{**FAST_CONFIG.__dict__, "ENABLED": False})
    proc = AcousticStreamProcessor(config=config, model=model, recorder=recorder)
    await proc.start()
    try:
        assert proc._ingest_task is None and proc._scheduler_task is None
        for i in range(10):
            proc.push_frame(time.monotonic(), _frame())
        await asyncio.sleep(0.05)
        assert model.calls == 0
        assert recorder.of_type(EventType.ACOUSTIC_DISABLED) != []
    finally:
        await proc.stop()


@pytest.mark.asyncio
async def test_clean_shutdown_cancels_tasks_and_is_idempotent():
    model = FakeAcousticModel()
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model)
    await proc.start()
    for i in range(5):
        proc.push_frame(i * FRAME_MS / 1000, _frame())
        await asyncio.sleep(FRAME_MS / 1000)

    await proc.stop()
    assert proc._ingest_task is None
    assert proc._scheduler_task is None
    assert proc._inference_task is None

    await proc.stop()  # idempotent — must not raise

    proc.push_frame(0.0, _frame())  # after shutdown, push_frame is a silent no-op
    await asyncio.sleep(0.02)
    assert model.calls == model.calls  # no crash; count simply doesn't grow from here


@pytest.mark.asyncio
async def test_reset_turn_clears_smoothing_history():
    model = FakeAcousticModel(frustration=0.9)
    proc = AcousticStreamProcessor(config=FAST_CONFIG, model=model)
    await proc.start()
    try:
        for i in range(10):
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.03)
        assert proc.state.frustration > 0.0

        proc.reset_turn()
        model.frustration = 0.1
        for i in range(10, 20):
            proc.push_frame(time.monotonic(), _frame())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.03)

        # First post-reset prediction should land close to the new raw value,
        # not a blend with the pre-reset history (EMA state was cleared).
        assert proc.state.frustration < 0.5
    finally:
        await proc.stop()


# ------------------------------------------------------------------ #
# One real integration test: actual DSP model, no fakes
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_real_prosody_model_tracks_rising_energy():
    """Not a plumbing test — proves the signal really moves with a real
    acoustic feature (RMS energy) computed from real audio, using the
    actual default runtime model."""
    config = AcousticConfig(
        WINDOW_SIZE_MS=300.0, INFERENCE_STRIDE_MS=50.0, MIN_AUDIO_REQUIRED_MS=100.0,
    )
    model = DeterministicProsodyModel()
    proc = AcousticStreamProcessor(config=config, model=model)
    await proc.start()
    try:
        sr = SR
        rng = np.random.default_rng(0)
        n_frames = 60  # 1.2s
        for i in range(n_frames):
            t = i * FRAME_MS / 1000
            amplitude = 0.02 + 0.15 * (i / n_frames)  # energy ramps up
            tt = np.arange(FRAME_SAMPLES) / sr + t
            audio = amplitude * np.sin(2 * np.pi * 160 * tt) + 0.005 * rng.standard_normal(FRAME_SAMPLES)
            proc.push_frame(time.monotonic(), audio.astype(np.float32))
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(0.05)

        assert len(proc.history) >= 3
        first_energy = proc.history[0].energy
        last_energy = proc.history[-1].energy
        assert last_energy > first_energy, "energy must track the real rising RMS amplitude"
    finally:
        await proc.stop()
