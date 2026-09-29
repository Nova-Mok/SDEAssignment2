"""Tests the one real conversational decision the acoustic signal drives
(see acoustic/policy.py + README "Using acoustic information"): while
smoothed frustration is acoustically high and confidence is sufficient,
BackchannelEngine must back off — longer cooldown, neutral-only phrases —
without any other change to its behavior. Uses the SAME fakes and
ManualClock as test_backchannel_engine.py; nothing here touches real audio
or a real model, only `BackchannelEngine.on_expression_update()`.
"""
from __future__ import annotations

import asyncio

from acoustic.config import AcousticConfig
from acoustic.types import ExpressionState
from backchannel.config import BackchannelConfig
from backchannel.engine import BackchannelEngine
from backchannel.events import EventType
from backchannel.instrumentation import InMemoryRecorder

from .fakes import FakeAudioProvider, ManualClock

BC_CONFIG = BackchannelConfig(
    MIN_SPEECH_DURATION_MS=1000.0,
    BACKCHANNEL_COOLDOWN_MS=1000.0,
    MAX_EOT_PROBABILITY=0.55,
    MIN_TIME_BETWEEN_DECISIONS_MS=100.0,
    MAX_BACKCHANNELS_PER_TURN=5,
    CANCEL_ON_EOT=True,
    SUPPRESS_NEAR_EOT_MS=300.0,
    BACKCHANNEL_AUDIBLE_GRACE_MS=50.0,
    EOT_SILENCE_WINDOW_MS=3000.0,
    EOT_LONG_TURN_MS=10000.0,
)
ACOUSTIC_CONFIG = AcousticConfig(
    FRUSTRATION_HIGH_THRESHOLD=0.7,
    FRUSTRATION_POLICY_MIN_CONFIDENCE=0.6,
    COOLDOWN_MULTIPLIER_ON_FRUSTRATION=3.0,
    NEUTRAL_ONLY_PHRASES=("mm-hmm", "okay"),
)


def _make_engine(clock: ManualClock, provider: FakeAudioProvider, recorder: InMemoryRecorder) -> BackchannelEngine:
    return BackchannelEngine(
        config=BC_CONFIG, audio_provider=provider, recorder=recorder, clock=clock,
        acoustic_config=ACOUSTIC_CONFIG,
    )


async def test_without_expression_update_behavior_is_unchanged():
    clock = ManualClock(start=0.0)
    provider = FakeAudioProvider()
    recorder = InMemoryRecorder()
    engine = _make_engine(clock, provider, recorder)

    engine.on_user_state_changed("speaking", now=clock())
    clock.advance(1.1)
    engine.tick(now=clock())
    await asyncio.sleep(0)

    selected = recorder.of_type(EventType.BACKCHANNEL_SELECTED)
    assert len(selected) == 1
    assert selected[0].metadata["acousticPolicyReason"] == "normal"
    assert selected[0].metadata["effectiveCooldownMs"] == BC_CONFIG.BACKCHANNEL_COOLDOWN_MS
    await engine.shutdown()


async def test_high_frustration_extends_cooldown_and_restricts_to_neutral_phrases():
    clock = ManualClock(start=0.0)
    # A short real play_duration_s so the FakeAudioProvider's playback (tied to
    # REAL time, unlike the rest of this test) actually finishes before the
    # second tick() below — otherwise tick()'s "one backchannel at a time"
    # guard would suppress it for the wrong reason.
    provider = FakeAudioProvider(phrases=["mm-hmm", "okay", "right", "yeah", "got-it"], play_duration_s=0.01)
    recorder = InMemoryRecorder()
    engine = _make_engine(clock, provider, recorder)

    engine.on_expression_update(ExpressionState(frustration=0.85, uncertainty=0.2, energy=0.6, confidence=0.9))

    engine.on_user_state_changed("speaking", now=clock())
    clock.advance(1.1)
    engine.tick(now=clock())  # first backchannel of the turn — always allowed regardless of cooldown
    await asyncio.sleep(0.03)  # let the fake playback actually finish (real time)

    first = recorder.of_type(EventType.BACKCHANNEL_SELECTED)
    assert len(first) == 1
    assert first[0].metadata["phrase"] in ACOUSTIC_CONFIG.NEUTRAL_ONLY_PHRASES
    assert first[0].metadata["acousticPolicyReason"] == "high_frustration_neutral_only"
    assert first[0].metadata["effectiveCooldownMs"] == BC_CONFIG.BACKCHANNEL_COOLDOWN_MS * 3.0

    # Advance past the NORMAL cooldown (1000ms) but not the acoustically
    # extended one (3000ms) — a second backchannel must still be suppressed.
    engine._last_backchannel_at = clock()
    clock.advance(1.5)
    engine.tick(now=clock())
    suppressed = recorder.of_type(EventType.BACKCHANNEL_SUPPRESSED)
    assert any("cooldown_active" in e.metadata["reason"] for e in suppressed)
    await engine.shutdown()


async def test_high_frustration_with_low_confidence_does_not_change_behavior():
    clock = ManualClock(start=0.0)
    provider = FakeAudioProvider()
    recorder = InMemoryRecorder()
    engine = _make_engine(clock, provider, recorder)

    engine.on_expression_update(ExpressionState(frustration=0.95, confidence=0.1))
    engine.on_user_state_changed("speaking", now=clock())
    clock.advance(1.1)
    engine.tick(now=clock())
    await asyncio.sleep(0)

    selected = recorder.of_type(EventType.BACKCHANNEL_SELECTED)
    assert selected[0].metadata["acousticPolicyReason"] == "insufficient_confidence"
    assert selected[0].metadata["effectiveCooldownMs"] == BC_CONFIG.BACKCHANNEL_COOLDOWN_MS
    await engine.shutdown()


def test_expression_update_is_recorded_as_its_own_event():
    clock = ManualClock(start=0.0)
    provider = FakeAudioProvider()
    recorder = InMemoryRecorder()
    engine = _make_engine(clock, provider, recorder)

    engine.on_expression_update(ExpressionState(frustration=0.4, uncertainty=0.3, energy=0.5, confidence=0.7))

    events = recorder.of_type(EventType.ACOUSTIC_EXPRESSION_UPDATED)
    assert len(events) == 1
    assert events[0].metadata["frustration"] == 0.4
