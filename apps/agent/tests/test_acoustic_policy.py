from __future__ import annotations

from acoustic.config import AcousticConfig
from acoustic.policy import evaluate_acoustic_policy
from acoustic.types import ExpressionState

CONFIG = AcousticConfig(
    FRUSTRATION_HIGH_THRESHOLD=0.7,
    FRUSTRATION_POLICY_MIN_CONFIDENCE=0.6,
    COOLDOWN_MULTIPLIER_ON_FRUSTRATION=2.0,
    NEUTRAL_ONLY_PHRASES=("mm-hmm", "okay"),
)


def _state(frustration: float, confidence: float) -> ExpressionState:
    return ExpressionState(frustration=frustration, confidence=confidence)


def test_no_state_yields_inert_default():
    decision = evaluate_acoustic_policy(None, CONFIG)
    assert decision.cooldown_multiplier == 1.0
    assert decision.allowed_phrases is None


def test_no_config_yields_inert_default():
    decision = evaluate_acoustic_policy(_state(0.9, 0.9), None)
    assert decision.cooldown_multiplier == 1.0
    assert decision.allowed_phrases is None


def test_disabled_config_yields_inert_default():
    disabled = AcousticConfig(ENABLED=False)
    decision = evaluate_acoustic_policy(_state(0.9, 0.9), disabled)
    assert decision.cooldown_multiplier == 1.0


def test_high_frustration_with_sufficient_confidence_triggers_policy():
    decision = evaluate_acoustic_policy(_state(0.85, 0.8), CONFIG)
    assert decision.cooldown_multiplier == 2.0
    assert decision.allowed_phrases == ("mm-hmm", "okay")
    assert decision.reason == "high_frustration_neutral_only"


def test_high_frustration_but_low_confidence_does_not_trigger_policy():
    decision = evaluate_acoustic_policy(_state(0.9, 0.2), CONFIG)
    assert decision.cooldown_multiplier == 1.0
    assert decision.allowed_phrases is None
    assert decision.reason == "insufficient_confidence"


def test_moderate_frustration_does_not_trigger_policy():
    decision = evaluate_acoustic_policy(_state(0.4, 0.9), CONFIG)
    assert decision.cooldown_multiplier == 1.0
    assert decision.reason == "normal"


def test_threshold_boundary_is_inclusive():
    decision = evaluate_acoustic_policy(_state(0.7, 0.6), CONFIG)
    assert decision.reason == "high_frustration_neutral_only"
