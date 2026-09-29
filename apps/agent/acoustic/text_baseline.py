"""A deliberately simple, independent text-sentiment baseline — used only
to demonstrate, in benchmarks/acoustic_vs_text.py, whether the acoustic
system provides information a text-only pipeline cannot see. This is NOT
meant to be a strong sentiment model; the entire point of Phase 7 is that
this baseline gets the SAME score for "Yeah, that's great" regardless of
whether it was said with genuine warmth or through gritted teeth, which is
precisely what a transcript-only signal is blind to.

Small hand-built lexicons, no external NLP dependency — keeps this
comparison honest and auditable (every matched word is visible in
`matched_terms`) rather than hiding behind an opaque model's own biases.
"""
from __future__ import annotations

from .types import TextBaselineScore

_POSITIVE_WORDS = {"great", "good", "awesome", "perfect", "love", "nice", "thanks", "thank", "wonderful", "yes"}
_FRUSTRATION_WORDS = {
    "ugh", "seriously", "ridiculous", "ugh", "ugh", "come on", "ridiculous", "annoying",
    "ugh", "again", "not again", "why", "frustrat", "stupid", "broken", "useless",
}
_UNCERTAINTY_WORDS = {
    "maybe", "i guess", "i think", "not sure", "kind of", "sort of", "um", "uh",
    "possibly", "perhaps", "i dunno", "dunno", "i don't know",
}


def score_text(text: str) -> TextBaselineScore:
    lowered = text.lower()
    matched: list[str] = []

    pos_hits = sum(1 for w in _POSITIVE_WORDS if w in lowered)
    frustration_hits = sum(1 for w in _FRUSTRATION_WORDS if w in lowered)
    uncertainty_hits = sum(1 for w in _UNCERTAINTY_WORDS if w in lowered)

    matched.extend(w for w in _POSITIVE_WORDS if w in lowered)
    matched.extend(w for w in _FRUSTRATION_WORDS if w in lowered)
    matched.extend(w for w in _UNCERTAINTY_WORDS if w in lowered)

    word_count = max(1, len(lowered.split()))
    positivity = min(1.0, pos_hits / max(1, word_count * 0.3))
    frustration_text = min(1.0, frustration_hits / max(1, word_count * 0.3))
    uncertainty_text = min(1.0, uncertainty_hits / max(1, word_count * 0.3))

    return TextBaselineScore(
        positivity=positivity,
        frustration_text=frustration_text,
        uncertainty_text=uncertainty_text,
        matched_terms=tuple(dict.fromkeys(matched)),  # de-dup, preserve order
    )
