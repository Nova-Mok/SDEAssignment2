"""Phase 7 — Acoustic vs. text baseline.

The transcript is IDENTICAL across all five `same_words_*` fixtures
(real Amazon Polly neural TTS, only prosody/SSML differs — see
scripts/generate_acoustic_fixtures.py). The text-sentiment baseline
(acoustic.text_baseline.score_text) therefore produces the exact same
score five times over, by construction — that's the point: a
transcript-only signal cannot see delivery at all.

The acoustic pipeline (default DeterministicProsodyModel) processes the
REAL audio for each variant independently and produces a different
frustration/uncertainty/energy reading per style, because it is actually
looking at rate/volume/pause characteristics the text never captures.

This does not claim the acoustic model "understands sarcasm" — sarcasm is
listed here because the assignment's example scenario includes it, and the
honest result (see printed output / README) is that sarcastic delivery is
the one style this system's signals do NOT clearly separate from
"uncertain", which is exactly the kind of limitation the assignment asks
us to report rather than paper over.

Usage: python -m benchmarks.acoustic_vs_text
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "apps" / "agent", REPO_ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from acoustic.config import AcousticConfig  # noqa: E402
from acoustic.mock_model import DeterministicProsodyModel  # noqa: E402
from acoustic.pipeline import AcousticStreamProcessor  # noqa: E402
from acoustic.text_baseline import score_text  # noqa: E402

from audio_io import load_fixture  # noqa: E402

SAME_WORDS_TRANSCRIPT = "Yeah, that's great."
STYLES = ["positive", "frustrated", "flat", "uncertain", "sarcastic"]


async def _run_acoustic(fixture_name: str) -> dict:
    audio, sample_rate = load_fixture(fixture_name)
    config = AcousticConfig(WINDOW_SIZE_MS=1000.0, INFERENCE_STRIDE_MS=100.0, MIN_AUDIO_REQUIRED_MS=300.0)
    model = DeterministicProsodyModel()
    proc = AcousticStreamProcessor(config=config, model=model)
    await proc.start()
    frame_len = int(sample_rate * 0.02)
    pos = 0
    while pos + frame_len <= audio.size:
        proc.push_frame(time.monotonic(), audio[pos : pos + frame_len])
        pos += frame_len
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.15)  # let the final scheduled inference land
    state = proc.state
    await proc.stop()
    return {
        "frustration": state.frustration,
        "uncertainty": state.uncertainty,
        "energy": state.energy,
        "confidence": state.confidence,
    }


async def main() -> None:
    text_score = score_text(SAME_WORDS_TRANSCRIPT)
    results = {"transcript": SAME_WORDS_TRANSCRIPT, "text_baseline": text_score.to_dict(), "acoustic_by_style": {}}

    print(f"Transcript (IDENTICAL for every row below): {SAME_WORDS_TRANSCRIPT!r}")
    print(f"Text baseline (computed once, from the transcript only): {text_score.to_dict()}\n")
    header = f"{'style':<12} {'ac_frustration':>15} {'ac_uncertainty':>15} {'ac_energy':>10} {'ac_confidence':>13}"
    print(header)
    print("-" * len(header))
    for style in STYLES:
        acoustic = await _run_acoustic(f"same_words_{style}")
        results["acoustic_by_style"][style] = acoustic
        print(
            f"{style:<12} {acoustic['frustration']:>15.3f} {acoustic['uncertainty']:>15.3f} "
            f"{acoustic['energy']:>10.3f} {acoustic['confidence']:>13.3f}"
        )

    frustrations = [v["frustration"] for v in results["acoustic_by_style"].values()]
    results["acoustic_frustration_range"] = max(frustrations) - min(frustrations)

    out_path = Path(__file__).resolve().parent.parent / "data" / "acoustic_vs_text.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(
        f"\nText signal: IDENTICAL across all 5 rows (by construction — same transcript).\n"
        f"Acoustic frustration range across styles: {results['acoustic_frustration_range']:.3f} "
        f"(0 = no discrimination, 1 = maximum).\n"
        f"Wrote {out_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
