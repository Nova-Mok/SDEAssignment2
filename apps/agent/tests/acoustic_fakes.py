"""Test doubles for the acoustic pipeline — mirrors tests/fakes.py's
philosophy (no real network/model weights anywhere in the unit test suite)
but kept in its own module since it's specific to acoustic/, not
backchannel/."""
from __future__ import annotations

import time

from acoustic.model import ModelUnavailableError
from acoustic.types import ModelInfo, RawModelOutput

__all__ = ["FakeAcousticModel", "ModelUnavailableError"]


class FakeAcousticModel:
    """Deterministic, controllable stand-in for a real `AcousticModel`:
    fixed output values, an optional blocking delay (runs inside
    `asyncio.to_thread` in real use, so `time.sleep` here is the right
    thing to simulate "slow inference"), and optional failure injection."""

    def __init__(
        self,
        frustration: float = 0.5,
        uncertainty: float = 0.3,
        confidence: float = 0.8,
        delay_s: float = 0.0,
        fail: bool = False,
        fail_with: type[Exception] = RuntimeError,
    ):
        self.frustration = frustration
        self.uncertainty = uncertainty
        self.confidence = confidence
        self.delay_s = delay_s
        self.fail = fail
        self.fail_with = fail_with
        self.calls = 0
        self._info = ModelInfo(
            name="fake-acoustic-model",
            version="fake-v1",
            kind="mock_dsp",
            sample_rate=16000,
            min_duration_ms=100.0,
            device="cpu",
            is_simulated=True,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, audio, sample_rate: int) -> RawModelOutput:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise self.fail_with("simulated acoustic model failure")
        return RawModelOutput(frustration=self.frustration, uncertainty=self.uncertainty, confidence=self.confidence)
