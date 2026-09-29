from __future__ import annotations

from acoustic.smoothing import ConfidenceWeightedEma, build_smoother


def test_first_sample_is_returned_unsmoothed():
    ema = ConfidenceWeightedEma(alpha=0.35)
    out = ema.update({"frustration": 0.8, "uncertainty": 0.2, "energy": 0.5}, confidence=0.9)
    assert out == {"frustration": 0.8, "uncertainty": 0.2, "energy": 0.5}


def test_smoothing_dampens_a_sudden_jump():
    ema = ConfidenceWeightedEma(alpha=0.35)
    ema.update({"frustration": 0.1, "uncertainty": 0.1, "energy": 0.1}, confidence=0.9)
    out = ema.update({"frustration": 0.9, "uncertainty": 0.9, "energy": 0.9}, confidence=0.9)
    assert 0.1 < out["frustration"] < 0.9, "a single new sample must not fully overwrite prior state"


def test_low_confidence_sample_moves_state_less_than_high_confidence():
    low = ConfidenceWeightedEma(alpha=0.35)
    high = ConfidenceWeightedEma(alpha=0.35)
    for ema in (low, high):
        ema.update({"frustration": 0.0, "uncertainty": 0.0, "energy": 0.0}, confidence=1.0)

    low_out = low.update({"frustration": 1.0, "uncertainty": 1.0, "energy": 1.0}, confidence=0.1)
    high_out = high.update({"frustration": 1.0, "uncertainty": 1.0, "energy": 1.0}, confidence=1.0)
    assert low_out["frustration"] < high_out["frustration"]


def test_reset_clears_state():
    ema = ConfidenceWeightedEma(alpha=0.35)
    ema.update({"frustration": 0.9, "uncertainty": 0.9, "energy": 0.9}, confidence=1.0)
    ema.reset()
    out = ema.update({"frustration": 0.1, "uncertainty": 0.1, "energy": 0.1}, confidence=1.0)
    assert out == {"frustration": 0.1, "uncertainty": 0.1, "energy": 0.1}


def test_build_smoother_rejects_unknown_method():
    import pytest

    with pytest.raises(ValueError):
        build_smoother("moving_average", alpha=0.3)
