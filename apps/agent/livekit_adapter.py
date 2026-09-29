"""The only file that bridges `backchannel/` (framework-agnostic) to real
LiveKit Agents objects. Responsibilities:

1. `LiveKitCachedAudioProvider` — plays a cached clip via the public
   `BackgroundAudioPlayer`, which publishes its own independent audio
   track. Wrapping its `PlayHandle` behind our own `PlaybackHandle`-shaped
   interface is what lets `BackchannelEngine` stay ignorant of LiveKit
   entirely while still driving real playback here.
2. `wire_session_events` — subscribes the three `AgentSession` events we
   established are public and sufficient (`user_state_changed`,
   `user_input_transcribed`, `agent_state_changed`) to the engine's three
   plain-data input methods. No other AgentSession internals are touched.
3. `AcousticTapAudioInput` / `wire_acoustic_tap` (Assignment 2) — the raw-
   audio side path. `session.input.audio` is a public, settable
   `AsyncIterable[rtc.AudioFrame]` chain (`livekit/agents/voice/io.py`);
   the tap wraps it, forwards every frame downstream UNCHANGED (STT sees
   byte-identical audio), and pushes a copy into the acoustic pipeline's
   queue on the side. This is what keeps acoustic inference a true side
   path rather than a second thing STT has to wait on — see README
   "Critical path".
4. `publish_expression_update` — pushes a compact JSON expression-state
   update to the browser over a LiveKit data-channel message (topic
   "acoustic"), so the live UI can render it without polling. See
   apps/web/app/live/page.tsx and README "Audio -> UI latency".
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents.voice import AgentSession, io
from livekit.agents.voice.background_audio import BackgroundAudioPlayer

from acoustic.types import ExpressionState
from backchannel.audio_provider import BackchannelAudioProvider
from backchannel.engine import BackchannelEngine

logger = logging.getLogger("livekit_adapter")


class _LiveKitPlayHandle:
    """Adapts BackgroundAudioPlayer's real PlayHandle to the
    stop()/done()/wait() shape BackchannelEngine expects."""

    def __init__(self, real_handle):
        self._real = real_handle

    def stop(self) -> None:
        self._real.stop()

    def done(self) -> bool:
        return self._real.done()

    async def wait(self) -> None:
        await self._real.wait_for_playout()


class LiveKitCachedAudioProvider(BackchannelAudioProvider):
    """Plays a pre-rendered clip through the room's real background-audio
    track. This is the production path — see README "Audio strategy" for
    why cached clips are preferred over on-demand TTS here."""

    def __init__(self, bg_audio: BackgroundAudioPlayer, clip_dir: Path):
        self._bg_audio = bg_audio
        self._clip_dir = Path(clip_dir)
        self._phrases = sorted(p.stem for p in self._clip_dir.glob("*.wav"))
        if not self._phrases:
            raise RuntimeError(f"no cached clips found in {self._clip_dir} — run scripts/generate_cached_clips.py first")

    def available_phrases(self) -> list[str]:
        return list(self._phrases)

    async def play(self, phrase: str, *, decision_id: int, turn_id: int):
        real_handle = self._bg_audio.play(str(self._clip_dir / f"{phrase}.wav"))
        return _LiveKitPlayHandle(real_handle)


def wire_session_events(session: AgentSession, engine: BackchannelEngine, acoustic_processor=None) -> None:
    def on_user_state_changed(ev) -> None:
        if acoustic_processor is not None and ev.new_state == "speaking":
            # A new turn starting: don't let the previous turn's smoothed
            # expression bleed into this one (see acoustic/pipeline.py
            # `reset_turn`). This is the one place engine.py and the
            # acoustic pipeline share a "turn" concept, and it lives here —
            # the bridging file — rather than in either framework-agnostic
            # module.
            acoustic_processor.reset_turn()
        engine.on_user_state_changed(ev.new_state)

    def on_user_input_transcribed(ev) -> None:
        engine.on_transcript(ev.transcript, ev.is_final)

    def on_agent_state_changed(ev) -> None:
        engine.on_agent_state_changed(ev.new_state)

    session.on("user_state_changed", on_user_state_changed)
    session.on("user_input_transcribed", on_user_input_transcribed)
    session.on("agent_state_changed", on_agent_state_changed)


class AcousticTapAudioInput(io.AudioInput):
    """Sits in front of whatever `session.input.audio` currently is (the
    real microphone track once RoomIO attaches it) and forwards every frame
    downstream byte-for-byte unchanged. The only side effect is a
    non-blocking callback — if that callback raises, forwarding still
    succeeds; acoustic bugs can never affect what STT hears."""

    def __init__(self, source: io.AudioInput, on_frame: Callable[[float, np.ndarray], None]):
        super().__init__(label="acoustic_tap", source=source)
        self._on_frame = on_frame

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self.source.__anext__()
        try:
            samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            self._on_frame(time.monotonic(), samples)
        except Exception:  # noqa: BLE001 - the shared audio path must never see an exception from here
            logger.exception("acoustic tap failed to process a frame; forwarding continues unaffected")
        return frame


def wire_acoustic_tap(session: AgentSession, processor) -> bool:
    """Install the tap. Must be called AFTER `await session.start(...)` —
    before that, RoomIO hasn't attached a real audio source yet and
    `session.input.audio` may still be None. Returns False (and leaves the
    session untouched) if there's no audio source to tap — including a
    session object that doesn't even expose `.input` (a test double, or a
    future AgentSession shape we didn't anticipate) — which the caller
    treats the same as "model unavailable": acoustic signal simply doesn't
    run, the call proceeds normally. This function must never raise; wiring
    an optional side path is never worth risking the real session setup."""
    try:
        audio_in = session.input.audio
    except AttributeError:
        logger.warning("acoustic: session has no '.input.audio' — cannot wire the tap; acoustic signal will not update")
        return False
    if audio_in is None:
        logger.warning("acoustic: session.input.audio is None — cannot wire the tap; acoustic signal will not update")
        return False
    session.input.audio = AcousticTapAudioInput(audio_in, processor.push_frame)
    return True


def publish_expression_update(room, state: ExpressionState) -> None:
    """Push one expression-state update to the browser over a LiveKit data
    message (topic "acoustic") — real-time, no polling, using the same
    room connection everything else already uses. Fire-and-forget: a
    publish failure (e.g. participant briefly disconnected) must not affect
    the call, so this schedules the coroutine and never awaits it inline
    from the acoustic pipeline's callback (which must stay synchronous)."""
    import asyncio

    payload = json.dumps({**state.to_dict(), "sentAt": time.time()}).encode("utf-8")

    async def _publish() -> None:
        try:
            await room.local_participant.publish_data(payload, reliable=True, topic="acoustic")
        except Exception:  # noqa: BLE001 - never let a transport hiccup affect the call
            logger.warning("acoustic: failed to publish expression update to room", exc_info=True)

    asyncio.create_task(_publish())
