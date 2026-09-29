"""Streaming acoustic-expression system.

See README "Streaming design" for the full architecture. Quick map:

  types.py         strongly-typed data model (predictions, windows, state)
  config.py        all tunables, env/JSON overridable (ACOUSTIC_*)
  model.py         the AcousticModel Protocol every implementation satisfies
  mock_model.py    default runtime model: fast, deterministic DSP/prosody
  real_model.py    real pretrained wav2vec2 emotion-recognition adapter
  features.py      shared low-level feature extraction (energy, pitch, ...)
  smoothing.py     confidence-weighted EMA
  policy.py        the one backchannel decision the acoustic signal drives
  text_baseline.py independent lexicon baseline, for the acoustic-vs-text comparison
  pipeline.py       AcousticStreamProcessor — the streaming engine itself
  factory.py       config -> constructed model (or None), isolating failures
"""
