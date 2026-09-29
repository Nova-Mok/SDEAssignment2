"""One-off script (mirrors scripts/generate_cached_clips.py's pattern):
renders the acoustic benchmark's audio fixtures via real AWS Polly neural
TTS, using SSML <prosody>/<break>/<emphasis> tags to vary DELIVERY while
holding the WORDS constant for the "same words, different delivery"
scenario — built on genuine synthesized speech, not hand-rolled sine-wave
synthesis.

IMPORTANT: these are neural-TTS speech, not human recordings. Every place
they're used (benchmarks/acoustic_scenarios.py, the README) says so. Run
once; output is committed to benchmarks/audio_fixtures/ so the benchmark
itself needs neither AWS credentials nor network access afterwards.

Known limitations, documented rather than hidden:

1. This AWS account/region has zero hi-IN Polly voices
   (`describe_voices(LanguageCode="hi-IN")` returns an empty list here).
   The "hindi_hinglish" fixture below is therefore an English neural voice
   reading Hinglish/code-switched text — not real Hindi speech — which is
   itself a realistic accent/language-mismatch edge case, and is called
   out explicitly in the README's "accents and languages" answer rather
   than papered over.
2. Empirically verified against this account (see README "Benchmark
   scenarios"): Polly's NEURAL engine rejects SSML
   `<prosody pitch="...">` and any `<emphasis>` level for this voice
   ("Unsupported Neural feature") — a real Polly platform constraint, not
   a corner cut here. Delivery variation below therefore uses only
   `rate`, `volume`, and `<break>` timing, which the neural engine does
   support and which still produces clearly distinct, audibly different
   deliveries of identical text.

Usage: python -m scripts.generate_acoustic_fixtures
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

import aws_providers  # noqa: F401 - resolves AWS creds via the "wavy" profile as an import side effect
import boto3

SAMPLE_RATE = 16000
OUT_DIR = AGENT_DIR.parent.parent / "benchmarks" / "audio_fixtures"
VOICE = "Matthew"
ENGINE = "neural"

_session = boto3.Session(profile_name="wavy", region_name="us-east-1")
_polly = _session.client("polly")


def _synthesize_pcm(ssml: str, voice: str = VOICE, engine: str = ENGINE) -> bytes:
    resp = _polly.synthesize_speech(
        Text=ssml, TextType="ssml", OutputFormat="pcm", VoiceId=voice, Engine=engine, SampleRate=str(SAMPLE_RATE),
    )
    return resp["AudioStream"].read()


def _write_wav(path: Path, pcm: bytes) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return len(pcm) / 2 / SAMPLE_RATE * 1000  # duration ms


def _silence_ms(ms: float) -> bytes:
    n_samples = int(SAMPLE_RATE * ms / 1000)
    return b"\x00\x00" * n_samples


# --------------------------------------------------------------------- #
# 1. Same words, different delivery — the transcript is IDENTICAL across
#    all five; only prosody (rate/pitch/volume/timing) changes.
# --------------------------------------------------------------------- #
SAME_WORDS_TRANSCRIPT = "Yeah, that's great."
SAME_WORDS_SSML = {
    "positive": '<speak><prosody rate="108%" volume="loud">Yeah, that\'s great.</prosody></speak>',
    "frustrated": '<speak><prosody rate="125%" volume="x-loud">Yeah<break time="150ms"/>, that\'s great.</prosody></speak>',
    "flat": '<speak><prosody rate="85%" volume="soft">Yeah. That\'s great.</prosody></speak>',
    "uncertain": '<speak><prosody rate="90%" volume="medium">Yeah<break time="400ms"/>, that\'s great<break time="200ms"/>?</prosody></speak>',
    "sarcastic": '<speak><prosody rate="90%" volume="medium">Yeah, that\'s<break time="350ms"/> <prosody rate="72%">great</prosody>.</prosody></speak>',
}

# --------------------------------------------------------------------- #
# 2. Increasing frustration — one continuous turn, four escalating steps.
#    Segment boundaries are recorded in the manifest so the benchmark can
#    correlate "acoustic signal at time T" with "intended intensity step".
# --------------------------------------------------------------------- #
FRUSTRATION_STEPS = [
    ('<speak><prosody rate="96%" volume="medium">Okay, so I filled out the form again.</prosody></speak>', "calm"),
    ('<speak><prosody rate="105%" volume="medium">I filled out the form again, like you asked.</prosody></speak>', "mild"),
    ('<speak><prosody rate="115%" volume="loud">I have filled out this form three times now.</prosody></speak>', "rising"),
    ('<speak><prosody rate="128%" volume="x-loud">This is the fourth time I am filling out this form.</prosody></speak>', "high"),
]

# --------------------------------------------------------------------- #
# 3. Hesitant speaker — pauses, fillers, uneven rate.
# --------------------------------------------------------------------- #
HESITANT_SSML = (
    '<speak><prosody rate="88%" volume="medium">Um<break time="350ms"/>, I think<break time="300ms"/> '
    "maybe it's the<break time=\"250ms\"/> second option? I'm not totally sure.</prosody></speak>"
)

# --------------------------------------------------------------------- #
# 4. High-energy speaker — fast, loud, animated.
# --------------------------------------------------------------------- #
HIGH_ENERGY_SSML = (
    '<speak><prosody rate="132%" volume="x-loud">'
    "This is amazing, I can't believe it actually worked, let's go!"
    "</prosody></speak>"
)

# --------------------------------------------------------------------- #
# 5. Long conversation — expression drifts within one turn: calm, then
#    uncertain, then frustrated. Segment boundaries recorded like #2.
# --------------------------------------------------------------------- #
LONG_CONVERSATION_STEPS = [
    ('<speak><prosody rate="97%" volume="medium">So I was looking at the invoice from last month.</prosody></speak>', "calm"),
    ('<speak><prosody rate="90%" volume="medium">I think<break time="300ms"/> there might be a<break time="250ms"/> duplicate charge? I\'m not sure.</prosody></speak>', "uncertain"),
    ('<speak><prosody rate="118%" volume="x-loud">I\'ve been charged twice for the same thing.</prosody></speak>', "frustrated"),
]

# --------------------------------------------------------------------- #
# 6. Hindi/Hinglish (code-switched) — see module docstring for the
#    voice-availability caveat.
# --------------------------------------------------------------------- #
HINGLISH_SSML = '<speak><prosody rate="95%">Yaar, ye form phir se submit karna padega? Bahut frustrating hai.</prosody></speak>'
HINGLISH_TRANSCRIPT = "Yaar, ye form phir se submit karna padega? Bahut frustrating hai."


def main() -> None:
    manifest: dict = {"sample_rate": SAMPLE_RATE, "voice": VOICE, "engine": ENGINE, "fixtures": {}}

    for style, ssml in SAME_WORDS_SSML.items():
        pcm = _synthesize_pcm(ssml)
        duration_ms = _write_wav(OUT_DIR / f"same_words_{style}.wav", pcm)
        manifest["fixtures"][f"same_words_{style}"] = {
            "transcript": SAME_WORDS_TRANSCRIPT, "style": style, "ssml": ssml,
            "durationMs": duration_ms, "kind": "same_words_different_delivery",
        }
        print(f"same_words_{style}: {duration_ms:.0f}ms")

    pcm_segments, offsets_ms, running_ms = [], [], 0.0
    for ssml, label in FRUSTRATION_STEPS:
        pcm = _synthesize_pcm(ssml)
        offsets_ms.append({"label": label, "startMs": running_ms})
        pcm_segments.append(pcm)
        running_ms += len(pcm) / 2 / SAMPLE_RATE * 1000
        pcm_segments.append(_silence_ms(300))
        running_ms += 300
    full_pcm = b"".join(pcm_segments)
    duration_ms = _write_wav(OUT_DIR / "increasing_frustration.wav", full_pcm)
    manifest["fixtures"]["increasing_frustration"] = {
        "kind": "increasing_frustration", "durationMs": duration_ms, "segments": offsets_ms,
    }
    print(f"increasing_frustration: {duration_ms:.0f}ms, segments={offsets_ms}")

    for name, ssml, transcript, kind in [
        ("hesitant_speaker", HESITANT_SSML, "Um, I think maybe it's the second option? I'm not totally sure.", "hesitant_speaker"),
        ("high_energy_speaker", HIGH_ENERGY_SSML, "This is amazing, I can't believe it actually worked, let's go!", "high_energy_speaker"),
        ("hindi_hinglish", HINGLISH_SSML, HINGLISH_TRANSCRIPT, "hindi_hinglish"),
    ]:
        pcm = _synthesize_pcm(ssml)
        duration_ms = _write_wav(OUT_DIR / f"{name}.wav", pcm)
        manifest["fixtures"][name] = {"transcript": transcript, "ssml": ssml, "durationMs": duration_ms, "kind": kind}
        print(f"{name}: {duration_ms:.0f}ms")

    pcm_segments, offsets_ms, running_ms = [], [], 0.0
    for ssml, label in LONG_CONVERSATION_STEPS:
        pcm = _synthesize_pcm(ssml)
        offsets_ms.append({"label": label, "startMs": running_ms})
        pcm_segments.append(pcm)
        running_ms += len(pcm) / 2 / SAMPLE_RATE * 1000
        pcm_segments.append(_silence_ms(400))
        running_ms += 400
    full_pcm = b"".join(pcm_segments)
    duration_ms = _write_wav(OUT_DIR / "long_conversation.wav", full_pcm)
    manifest["fixtures"]["long_conversation"] = {
        "kind": "long_conversation", "durationMs": duration_ms, "segments": offsets_ms,
    }
    print(f"long_conversation: {duration_ms:.0f}ms, segments={offsets_ms}")

    manifest["note"] = (
        "All fixtures are Amazon Polly neural-engine TTS synthesized from SSML "
        "(rate/pitch/volume/break/emphasis tags), NOT human recordings. "
        "hindi_hinglish uses an English (Matthew) neural voice reading "
        "Hinglish text — this AWS account/region has no hi-IN Polly voice "
        "available (describe_voices returned none); see README 'accents and "
        "languages'."
    )
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote manifest to {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
