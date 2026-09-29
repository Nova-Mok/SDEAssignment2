"""Prediction smoothing.

Chosen method: confidence-weighted exponential moving average (EMA). Only
one method is implemented, per the assignment's "choose one and explain
why":

- A plain moving average over the last N raw predictions would need to
  buffer N predictions before its output means anything, adding N *
  INFERENCE_STRIDE_MS of latency on top of the detection latency the
  pipeline is already paying for window-fill. That directly fights the
  "smoothing must not introduce excessive latency" requirement.
- EMA is O(1) state (a single running value per signal), reacts to every
  new sample immediately (no buffering delay), and its effective
  half-life is a single tunable (SMOOTHING_ALPHA) rather than a window
  length — cheaper and simpler to reason about.
- Making it confidence-aware (scaling the effective alpha by the new
  sample's confidence) means a low-confidence prediction — e.g. from a
  window that was mostly silence — nudges the smoothed state gently
  instead of causing a visible jump, without needing a second smoothing
  pass or a hard confidence gate that would just drop information.

See benchmarks/acoustic_latency_bench.py's `smoothing_impact` section for
a measured before/after example of raw vs. smoothed trajectories.
"""
from __future__ import annotations

from typing import Protocol

_SIGNAL_KEYS = ("frustration", "uncertainty", "energy")


class Smoother(Protocol):
    def update(self, raw: dict[str, float], confidence: float) -> dict[str, float]:
        """Fold in one new raw sample, return the updated smoothed values."""
        ...

    def reset(self) -> None:
        """Clear all state — called at the start of a new user turn so one
        turn's expression never bleeds smoothing history into the next."""
        ...


class ConfidenceWeightedEma(Smoother):
    def __init__(self, alpha: float = 0.35, min_effective_alpha: float = 0.05):
        self._alpha = alpha
        self._min_effective_alpha = min_effective_alpha
        self._state: dict[str, float] | None = None

    def update(self, raw: dict[str, float], confidence: float) -> dict[str, float]:
        effective_alpha = max(self._min_effective_alpha, self._alpha * max(0.0, min(1.0, confidence)))
        if self._state is None:
            self._state = dict(raw)
            return dict(self._state)
        updated = {
            key: effective_alpha * raw.get(key, self._state[key]) + (1 - effective_alpha) * self._state[key]
            for key in _SIGNAL_KEYS
        }
        self._state = updated
        return dict(updated)

    def reset(self) -> None:
        self._state = None


def build_smoother(method: str, alpha: float) -> Smoother:
    if method != "ema":
        raise ValueError(f"unsupported SMOOTHING_METHOD: {method!r} (only 'ema' is implemented — see module docstring)")
    return ConfidenceWeightedEma(alpha=alpha)
