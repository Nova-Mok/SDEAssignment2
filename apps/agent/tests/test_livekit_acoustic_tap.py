"""Tests for the LiveKit-specific bridging code in livekit_adapter.py: the
audio tap must forward every frame byte-for-byte unchanged (STT's input can
never be affected by the acoustic side path) while also feeding a copy to
the acoustic pipeline; and the wiring helper must degrade gracefully for
any session shape that doesn't expose a tappable audio source.
"""
from __future__ import annotations

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice import io

from livekit_adapter import AcousticTapAudioInput, wire_acoustic_tap

SR = 16000
FRAME_SAMPLES = 320  # 20ms @ 16kHz


def _make_frame(fill_value: int) -> rtc.AudioFrame:
    samples = np.full(FRAME_SAMPLES, fill_value, dtype=np.int16)
    return rtc.AudioFrame(data=samples.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=FRAME_SAMPLES)


class _FakeSource(io.AudioInput):
    def __init__(self, frames: list[rtc.AudioFrame]):
        super().__init__(label="fake_source")
        self._frames = list(frames)

    async def __anext__(self) -> rtc.AudioFrame:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)


@pytest.mark.asyncio
async def test_tap_forwards_frames_unchanged():
    frames = [_make_frame(100), _make_frame(-200), _make_frame(0)]
    source = _FakeSource(list(frames))

    received: list[tuple[float, np.ndarray]] = []
    tap = AcousticTapAudioInput(source, lambda ts, samples: received.append((ts, samples)))

    forwarded = [await tap.__anext__() for _ in range(3)]

    assert [f.data for f in forwarded] == [f.data for f in frames]
    assert len(received) == 3
    # Second forwarded frame was filled with -200 (int16) -> normalized float32.
    assert received[1][1][0] == pytest.approx(-200 / 32768.0, abs=1e-6)


@pytest.mark.asyncio
async def test_tap_forwarding_survives_callback_exception():
    frames = [_make_frame(50)]
    source = _FakeSource(list(frames))

    def _boom(ts, samples):
        raise RuntimeError("acoustic bug")

    tap = AcousticTapAudioInput(source, _boom)
    forwarded = await tap.__anext__()
    assert forwarded.data == frames[0].data  # frame still delivered despite the callback raising


class _NoInputSession:
    """Mimics a session shape with no `.input` attribute at all — e.g. a
    test double, or a hypothetical future AgentSession variant."""


class _NoneAudioInputSession:
    class _Input:
        audio = None

    def __init__(self):
        self.input = self._Input()


class _RealAudioInputSession:
    def __init__(self, source: io.AudioInput):
        class _Input:
            pass

        self.input = _Input()
        self.input.audio = source


def test_wire_acoustic_tap_handles_missing_input_attribute_gracefully():
    assert wire_acoustic_tap(_NoInputSession(), processor=object()) is False


def test_wire_acoustic_tap_handles_none_audio_gracefully():
    assert wire_acoustic_tap(_NoneAudioInputSession(), processor=object()) is False


def test_wire_acoustic_tap_installs_tap_when_audio_source_present():
    source = _FakeSource([])

    class _Processor:
        def push_frame(self, ts, samples):
            pass

    session = _RealAudioInputSession(source)
    installed = wire_acoustic_tap(session, processor=_Processor())
    assert installed is True
    assert isinstance(session.input.audio, AcousticTapAudioInput)
