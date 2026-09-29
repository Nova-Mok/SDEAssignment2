from __future__ import annotations

from acoustic.text_baseline import score_text


def test_same_words_yield_same_score_regardless_of_intended_delivery():
    """The entire point of the text baseline: it cannot see delivery. Five
    deliveries of the same sentence must score identically here, even
    though a human listener (and the acoustic system) would tell them
    apart instantly — see benchmarks/acoustic_vs_text.py for the acoustic
    side of this same comparison."""
    text = "Yeah, that's great."
    scores = [score_text(text) for _ in range(5)]
    assert all(s == scores[0] for s in scores)


def test_positive_words_raise_positivity():
    neutral = score_text("okay")
    positive = score_text("that's great, thanks so much, awesome")
    assert positive.positivity > neutral.positivity


def test_frustration_words_raise_text_frustration():
    calm = score_text("that's great")
    frustrated = score_text("this is ridiculous, seriously, it's broken again")
    assert frustrated.frustration_text > calm.frustration_text


def test_hedge_words_raise_text_uncertainty():
    confident = score_text("that's great")
    hedging = score_text("maybe, i think, i guess, not sure")
    assert hedging.uncertainty_text > confident.uncertainty_text


def test_matched_terms_are_reported_for_auditability():
    result = score_text("i guess that's great")
    assert "i guess" in result.matched_terms
    assert "great" in result.matched_terms
