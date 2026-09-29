"""The one concrete conversational decision the acoustic signal is allowed
to change (per the assignment: "use the acoustic signal for at least one
real conversational decision... do not simply add it to the LLM prompt").

Policy (see README "Using acoustic information" for the full rationale and
how it's measured):

    IF smoothed frustration >= FRUSTRATION_HIGH_THRESHOLD
    AND confidence >= FRUSTRATION_POLICY_MIN_CONFIDENCE
    THEN:
      - multiply the backchannel cooldown by COOLDOWN_MULTIPLIER_ON_FRUSTRATION
        (fewer acknowledgements while frustration is high)
      - restrict phrase selection to NEUTRAL_ONLY_PHRASES (no casual "yeah"/
        "right" while the user sounds acoustically frustrated)

This module is intentionally free of any numpy/torch import — it is pure
arithmetic over plain dataclasses — so `backchannel/engine.py` can import it
directly without dragging the ML stack into a package whose own tests
(test_backchannel_engine.py) currently need no such dependency.
"""
from __future__ import annotations

import dataclasses

from .config import AcousticConfig
from .types import ExpressionState


@dataclasses.dataclass(frozen=True)
class PolicyDecision:
    cooldown_multiplier: float
    allowed_phrases: tuple[str, ...] | None  # None = no restriction
    reason: str


_DEFAULT_DECISION = PolicyDecision(cooldown_multiplier=1.0, allowed_phrases=None, reason="normal")


def evaluate_acoustic_policy(state: ExpressionState | None, config: AcousticConfig | None) -> PolicyDecision:
    if state is None or config is None or not config.ENABLED:
        return _DEFAULT_DECISION
    if state.confidence < config.FRUSTRATION_POLICY_MIN_CONFIDENCE:
        return dataclasses.replace(_DEFAULT_DECISION, reason="insufficient_confidence")
    if state.frustration >= config.FRUSTRATION_HIGH_THRESHOLD:
        return PolicyDecision(
            cooldown_multiplier=config.COOLDOWN_MULTIPLIER_ON_FRUSTRATION,
            allowed_phrases=config.NEUTRAL_ONLY_PHRASES,
            reason="high_frustration_neutral_only",
        )
    return _DEFAULT_DECISION
