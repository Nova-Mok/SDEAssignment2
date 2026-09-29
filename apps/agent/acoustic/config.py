"""All acoustic-pipeline thresholds live here, mirroring
`backchannel/config.py`'s pattern exactly: one frozen dataclass, every
number overridable via `ACOUSTIC_<FIELD>` env vars or a JSON file, so a
tenant, a benchmark run, and the live agent can each run a different policy
without a code change or a fork.

See README "Streaming design" for the reasoning behind the defaults below —
in short: WINDOW_SIZE_MS and INFERENCE_STRIDE_MS trade prediction stability
against detection latency (bigger window = smoother but slower to react);
MAX_QUEUE_SIZE/MAX_STALENESS_MS/INFERENCE_CONCURRENCY exist so a slow or
wedged model degrades the acoustic signal, never the call.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class AcousticConfig:
    # Master switch. False (or a model that fails to load) makes the entire
    # feature a no-op — zero tap, zero tasks, zero overhead. This is the
    # single flag "Customer C: acoustic analysis disabled" sets to True->False.
    ENABLED: bool = True

    # Which model implementation to construct. "mock" = the fast deterministic
    # DSP/prosody model (default: always meets the latency budget on CPU).
    # "real" = the pretrained wav2vec2 emotion-recognition adapter (see
    # acoustic/real_model.py) — measured slower on CPU; intended for
    # GPU-backed serving. See README "Model selection".
    MODEL_KIND: str = "mock"

    # --- Windowing / streaming -------------------------------------------------
    # How much trailing audio one prediction is computed from. Larger = more
    # stable (more pitch/energy cycles to average over), at the cost of a
    # window-fill delay that dominates true detection latency (see README
    # "Latency methodology" — this is the single biggest latency lever).
    WINDOW_SIZE_MS: float = 1500.0

    # How often a new prediction is attempted. Independent of WINDOW_SIZE_MS:
    # windows overlap (a new one is computed every stride using the trailing
    # WINDOW_SIZE_MS of audio), which is what makes this streaming rather
    # than "wait for the window, then wait again".
    INFERENCE_STRIDE_MS: float = 250.0

    # Below this much buffered audio, don't even attempt a prediction — the
    # window would be too short to mean anything. First prediction of a turn
    # therefore lands at MIN_AUDIO_REQUIRED_MS, not WINDOW_SIZE_MS.
    MIN_AUDIO_REQUIRED_MS: float = 500.0

    # --- Smoothing ---------------------------------------------------------
    # "ema" is the only implemented method (see acoustic/smoothing.py for why
    # a moving average was rejected: it adds a window-length of latency on
    # top of the detection latency we're already paying for; EMA is O(1) and
    # adds none). Kept as a string, not a bool, so a future second method is
    # a config change, not a code change at every call site.
    SMOOTHING_METHOD: str = "ema"

    # EMA weight on the newest sample, before confidence weighting.
    SMOOTHING_ALPHA: float = 0.35

    # Predictions below this confidence still update the UI (so "we're not
    # sure yet" is visible) but never drive the backchannel policy.
    CONFIDENCE_THRESHOLD: float = 0.35

    # --- Backpressure (see README "Backpressure") ---------------------------
    # Bound on the raw-frame ingestion queue between the (synchronous, must
    # never block) LiveKit audio tap and the async ingestion task. Full ->
    # oldest frame is dropped, never the newest — see pipeline.py.
    MAX_QUEUE_SIZE: int = 8

    # A window older than this (relative to "now" when inference would start)
    # is not worth computing — the speech it describes is no longer current.
    # Also the threshold used to cancel an in-flight inference that has been
    # running too long (see pipeline.py `_scheduler_loop`).
    MAX_STALENESS_MS: float = 800.0

    # Max number of concurrent inference calls in flight for one call. 1 is
    # correct for the default cadence-based scheduler (a new window is only
    # submitted once the previous inference finishes or is cancelled for
    # staleness) — kept as a field so a future batched/GPU server can raise it
    # without an interface change.
    INFERENCE_CONCURRENCY: int = 1

    # Hard cap on one inference call. Exceeding it cancels the call (freeing
    # the pipeline to try a fresher window) and is recorded as a failure, not
    # silently absorbed as extra latency.
    INFERENCE_TIMEOUT_MS: float = 500.0

    # --- Backchannel policy (see acoustic/policy.py + README "Using acoustic
    # information") ----------------------------------------------------------
    FRUSTRATION_HIGH_THRESHOLD: float = 0.70
    FRUSTRATION_POLICY_MIN_CONFIDENCE: float = 0.60
    COOLDOWN_MULTIPLIER_ON_FRUSTRATION: float = 2.0
    # Phrases considered "neutral" among the existing cached clips — used as
    # the only allowed acknowledgements while high frustration is active.
    NEUTRAL_ONLY_PHRASES: tuple[str, ...] = ("mm-hmm", "okay")

    @staticmethod
    def _coerce(raw: str, default):
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, tuple):
            return tuple(p.strip() for p in raw.split(",") if p.strip())
        return raw

    @classmethod
    def load(cls, json_path: str | os.PathLike | None = None, env_prefix: str = "ACOUSTIC_") -> "AcousticConfig":
        """Defaults, then JSON overrides, then env var overrides (env wins) —
        identical precedence to `BackchannelConfig.load()`."""
        overrides: dict = {}
        if json_path is not None and Path(json_path).exists():
            overrides.update(json.loads(Path(json_path).read_text()))
        base = dataclasses.replace(cls(), **overrides) if overrides else cls()

        env_overrides: dict = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(env_prefix + f.name)
            if raw is not None:
                env_overrides[f.name] = cls._coerce(raw, getattr(base, f.name))
        return dataclasses.replace(base, **env_overrides) if env_overrides else base


DEFAULT_CONFIG = AcousticConfig()
