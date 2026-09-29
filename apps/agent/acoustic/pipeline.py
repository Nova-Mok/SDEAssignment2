"""The streaming acoustic pipeline: turns a live stream of raw audio frames
into a smoothed, continuously-updated ExpressionState, without ever
touching (or being able to block) the core STT -> EOT -> LLM -> TTS path.

Architecture (see README "Streaming design" / "Critical path" for the full
picture):

    LiveKit audio tap (sync, non-blocking)
        -> bounded ingestion queue (drop-OLDEST on overflow)
        -> _ingest_loop (async): queue -> rolling audio buffer
        -> _scheduler_loop (async): fixed cadence (INFERENCE_STRIDE_MS)
             -> extract latest window
             -> asyncio.to_thread(model.predict, ...) with a timeout
             -> smoothing -> ExpressionState -> on_state_update callback

Both loops run as their own asyncio tasks, independent of the agent's main
session tasks. A crash, a timeout, or a completely unavailable model in
here can never propagate to or block the voice pipeline — every failure
path in this module ends in "log, record an event, keep going" (see the
`except Exception` clauses below), which is what makes this a true side
path rather than a second critical path in disguise. See
tests/test_acoustic_pipeline.py for behavioral proof of each of these
claims (slow inference, failed inference, queue overflow, stale-window
drop, disabled mode, clean shutdown).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import time
from collections.abc import Callable

import numpy as np

from backchannel.events import Event, EventType
from backchannel.instrumentation import EventRecorder

from . import features as _features
from .config import AcousticConfig
from .model import AcousticModel, ModelUnavailableError
from .smoothing import build_smoother
from .types import AcousticPrediction, ExpressionState, LatencyBreakdown, WindowInfo

logger = logging.getLogger("acoustic.pipeline")

SAMPLE_RATE = 16000


class AcousticStreamProcessor:
    def __init__(
        self,
        config: AcousticConfig,
        model: AcousticModel,
        recorder: EventRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
        run_id: str = "",
        scenario_id: str = "",
        on_state_update: Callable[[ExpressionState], None] | None = None,
    ) -> None:
        self._config = config
        self._model = model
        self._recorder = recorder
        self._clock = clock
        self._run_id = run_id
        self._scenario_id = scenario_id
        self._on_state_update = on_state_update

        self._queue: asyncio.Queue[tuple[float, np.ndarray]] = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)
        self._buffer: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._buffer_duration_ms = 0.0

        self._smoother = build_smoother(config.SMOOTHING_METHOD, config.SMOOTHING_ALPHA)
        self._state = ExpressionState()
        self._history: collections.deque[AcousticPrediction] = collections.deque(maxlen=200)

        self._ingest_task: asyncio.Task | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._inference_task: asyncio.Task | None = None
        self._inference_started_at: float | None = None

        self._shut_down = False
        self._dropped_frames = 0
        self._dropped_windows = 0

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> ExpressionState:
        return self._state

    @property
    def history(self) -> list[AcousticPrediction]:
        return list(self._history)

    @property
    def enabled(self) -> bool:
        return self._config.ENABLED and not self._shut_down

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def dropped_windows(self) -> int:
        return self._dropped_windows

    def set_run_context(self, run_id: str, scenario_id: str) -> None:
        """Rebind which run/scenario subsequent events are attributed to —
        used by benchmark harnesses (see replay_runner.run_one) that only
        learn the run_id after constructing the processor."""
        self._run_id = run_id
        self._scenario_id = scenario_id

    def reset_turn(self) -> None:
        """Called by the caller (e.g. on USER_SPEECH_START) so one turn's
        smoothing history never bleeds into the next turn's expression."""
        self._smoother.reset()

    def push_frame(self, frame_ts: float, samples: np.ndarray) -> None:
        """Non-blocking. Called on every ~20ms audio frame — from the
        LiveKit audio-tap coroutine in production (see
        livekit_adapter.AcousticTapAudioInput) or from a benchmark harness
        feeding a WAV fixture. `frame_ts` is the clock() time this frame
        arrived (treated as a proxy for when it was spoken — see README
        "Latency methodology" for the capture-delay caveat this implies).

        Must never await and must never raise into the caller: the
        production caller is on the audio-forwarding path shared with STT.
        """
        if not self.enabled:
            return
        try:
            self._queue.put_nowait((frame_ts, samples))
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()  # drop the OLDEST buffered frame, not the newest
            except asyncio.QueueEmpty:
                pass
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait((frame_ts, samples))
            self._dropped_frames += 1
            self._record(EventType.ACOUSTIC_QUEUE_OVERFLOW, metadata={"droppedFramesTotal": self._dropped_frames})
        except Exception:  # noqa: BLE001 - the audio-forwarding path must never see an exception from here
            logger.exception("acoustic pipeline: push_frame failed; frame dropped")

    async def start(self) -> None:
        if not self._config.ENABLED:
            self._record(EventType.ACOUSTIC_DISABLED, metadata={"reason": "config_disabled"})
            return
        if self._ingest_task is not None:
            return
        self._ingest_task = asyncio.create_task(self._ingest_loop())
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._shut_down = True
        for task in (self._scheduler_task, self._ingest_task, self._inference_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (self._scheduler_task, self._ingest_task, self._inference_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._scheduler_task = self._ingest_task = self._inference_task = None

    # ------------------------------------------------------------------ #
    # Ingestion: queue -> rolling buffer
    # ------------------------------------------------------------------ #

    async def _ingest_loop(self) -> None:
        try:
            while True:
                frame_ts, samples = await self._queue.get()
                self._buffer.append((frame_ts, samples))
                frame_ms = samples.size / SAMPLE_RATE * 1000
                self._buffer_duration_ms += frame_ms
                self._trim_buffer()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a bug here degrades acoustic processing, never the call
            logger.exception("acoustic pipeline: ingest loop crashed; acoustic signal will stop updating")

    def _trim_buffer(self) -> None:
        max_ms = self._config.WINDOW_SIZE_MS
        while self._buffer and self._buffer_duration_ms > max_ms:
            _, oldest = self._buffer[0]
            oldest_ms = oldest.size / SAMPLE_RATE * 1000
            if self._buffer_duration_ms - oldest_ms < max_ms * 0.5:
                # Keep some headroom rather than trimming to exactly max_ms
                # on every single frame — cheap hysteresis, avoids needless
                # deque churn without ever letting memory grow unbounded.
                break
            self._buffer.popleft()
            self._buffer_duration_ms -= oldest_ms

    def _latest_window(self) -> tuple[np.ndarray, float, float] | None:
        """Concatenate buffered frames into one array covering up to the
        last WINDOW_SIZE_MS. Returns (samples, window_start_ts, window_end_ts),
        where each buffered tuple's timestamp marks the END (arrival) of that
        chunk's audio, so a chunk's own start is `arrival_ts - duration`."""
        if not self._buffer:
            return None
        window_ms = self._config.WINDOW_SIZE_MS
        chunks: list[np.ndarray] = []
        total_ms = 0.0
        earliest_arrival_ts = None
        earliest_duration_ms = 0.0
        end_ts = self._buffer[-1][0]
        for arrival_ts, samples in reversed(self._buffer):
            chunks.append(samples)
            dur_ms = samples.size / SAMPLE_RATE * 1000
            total_ms += dur_ms
            earliest_arrival_ts = arrival_ts
            earliest_duration_ms = dur_ms
            if total_ms >= window_ms:
                break
        if not chunks or earliest_arrival_ts is None:
            return None
        audio = np.concatenate(list(reversed(chunks)))
        start_ts = earliest_arrival_ts - earliest_duration_ms / 1000
        return audio, start_ts, end_ts

    # ------------------------------------------------------------------ #
    # Scheduling: fixed cadence, latest-state, staleness cancellation
    # ------------------------------------------------------------------ #

    async def _scheduler_loop(self) -> None:
        interval_s = self._config.INFERENCE_STRIDE_MS / 1000
        try:
            while True:
                await asyncio.sleep(interval_s)
                self._maybe_reap_stale_inference()
                if self._inference_task is not None and not self._inference_task.done():
                    continue  # one inference at a time (default INFERENCE_CONCURRENCY==1) — never pile up

                if self._buffer_duration_ms < self._config.MIN_AUDIO_REQUIRED_MS:
                    continue

                window = self._latest_window()
                if window is None:
                    continue
                audio, start_ts, end_ts = window

                now = self._clock()
                staleness_ms = (now - end_ts) * 1000
                if staleness_ms > self._config.MAX_STALENESS_MS:
                    self._dropped_windows += 1
                    self._record(
                        EventType.ACOUSTIC_PREDICTION_STALE,
                        metadata={"stalenessMs": staleness_ms, "reason": "window_stale_before_submit"},
                    )
                    continue

                is_partial = self._buffer_duration_ms < self._config.WINDOW_SIZE_MS
                window_info = WindowInfo(
                    start_ts=start_ts, end_ts=end_ts,
                    duration_ms=audio.size / SAMPLE_RATE * 1000,
                    sample_count=audio.size, is_partial=is_partial,
                )
                self._record(EventType.ACOUSTIC_AUDIO_WINDOW_READY, metadata=window_info.to_dict())
                self._inference_started_at = now
                self._inference_task = asyncio.create_task(self._run_inference(audio, window_info))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - scheduler bugs degrade acoustic signal, never the call
            logger.exception("acoustic pipeline: scheduler loop crashed; acoustic signal will stop updating")

    def _maybe_reap_stale_inference(self) -> None:
        task, started_at = self._inference_task, self._inference_started_at
        if task is None or task.done() or started_at is None:
            return
        running_ms = (self._clock() - started_at) * 1000
        if running_ms > self._config.INFERENCE_TIMEOUT_MS:
            task.cancel()
            self._record(
                EventType.ACOUSTIC_PREDICTION_STALE,
                metadata={"runningMs": running_ms, "reason": "inference_exceeded_timeout_cancelled"},
            )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    async def _run_inference(self, audio: np.ndarray, window_info: WindowInfo) -> None:
        input_ready_ts = window_info.end_ts
        t_infer_start = self._clock()
        self._record(EventType.ACOUSTIC_INFERENCE_STARTED, metadata={"windowDurationMs": window_info.duration_ms})
        try:
            timeout_s = self._config.INFERENCE_TIMEOUT_MS / 1000
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._model.predict, audio, SAMPLE_RATE), timeout=timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._record(
                EventType.ACOUSTIC_INFERENCE_FAILED,
                metadata={"reason": "timeout", "timeoutMs": self._config.INFERENCE_TIMEOUT_MS},
            )
            return
        except ModelUnavailableError as exc:
            self._record(EventType.ACOUSTIC_MODEL_UNAVAILABLE, metadata={"error": repr(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - inference failure must never crash the call
            self._record(EventType.ACOUSTIC_INFERENCE_FAILED, metadata={"reason": "exception", "error": repr(exc)})
            return
        finally:
            if self._inference_task is asyncio.current_task():
                self._inference_task = None
                self._inference_started_at = None

        t_infer_end = self._clock()
        inference_latency_ms = (t_infer_end - t_infer_start) * 1000
        self._record(EventType.ACOUSTIC_INFERENCE_COMPLETED, metadata={"inferenceLatencyMs": inference_latency_ms})

        energy = _features.rms_energy(audio)
        raw_values = {"frustration": raw.frustration, "uncertainty": raw.uncertainty, "energy": energy}
        smoothed = self._smoother.update(raw_values, raw.confidence)

        latency = LatencyBreakdown(
            audio_wait_ms=max(0.0, (input_ready_ts - window_info.start_ts) * 1000),
            queue_wait_ms=max(0.0, (t_infer_start - input_ready_ts) * 1000),
            inference_latency_ms=inference_latency_ms,
            effective_detection_latency_ms=max(0.0, (t_infer_end - window_info.start_ts) * 1000),
        )

        prediction = AcousticPrediction(
            produced_at=t_infer_end,
            window=window_info,
            latency=latency,
            frustration=raw.frustration,
            uncertainty=raw.uncertainty,
            energy=energy,
            confidence=raw.confidence,
            model=self._model.info,
            smoothed_frustration=smoothed["frustration"],
            smoothed_uncertainty=smoothed["uncertainty"],
            smoothed_energy=smoothed["energy"],
        )
        self._history.append(prediction)

        self._state = ExpressionState(
            frustration=smoothed["frustration"],
            uncertainty=smoothed["uncertainty"],
            energy=smoothed["energy"],
            confidence=raw.confidence,
            updated_at=t_infer_end,
            sample_count=self._state.sample_count + 1,
            model_version=self._model.info.version,
            stale=False,
        )

        self._record(EventType.ACOUSTIC_PREDICTION_PRODUCED, metadata=prediction.to_dict())

        if self._on_state_update is not None:
            try:
                self._on_state_update(self._state)
            except Exception:  # noqa: BLE001 - a consumer bug must never break the pipeline itself
                logger.exception("acoustic pipeline: on_state_update callback raised")

    # ------------------------------------------------------------------ #

    def _record(self, event_type: EventType, metadata: dict | None = None) -> None:
        if self._recorder is None:
            return
        self._recorder.record(
            Event(
                event_type=event_type,
                timestamp=self._clock(),
                run_id=self._run_id,
                scenario_id=self._scenario_id,
                config="acoustic",
                source="acoustic",
                metadata=metadata or {},
            )
        )
