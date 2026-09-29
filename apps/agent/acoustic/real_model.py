"""The real acoustic model candidate: `superb/wav2vec2-base-superb-er`, a
wav2vec2-base encoder (self-supervised, pretrained directly on raw audio —
no transcript involvement anywhere in its architecture or training) fine-
tuned on IEMOCAP for 4-class categorical speech emotion recognition
(neutral / happy / angry / sad). See README "Model selection" for the full
justification, and PHASE2_MODEL_NOTES.md for how it was evaluated.

Measured on this machine (CPU, single-threaded, transformers 5.1.0 /
torch 2.10.0, Apple Silicon): ~95M params, load time ~6s, forward-pass
latency for a 1.5s window p50=118.9ms / p95=134.8ms. Against this system's
default INFERENCE_STRIDE_MS=250ms, that consumes roughly half the stride
budget on ONE call on ONE CPU core with nothing else running — not
practical as the default for many concurrent calls on commodity CPU, which
is why acoustic/mock_model.py is the default and this adapter is the
documented, GPU-serving-track alternative (see README "Model serving").

What this model's output actually represents, precisely (per the
assignment's "do not pretend capabilities that cannot be verified"):
  - `frustration` = the classifier's P(angry). It is a genuine model
    prediction, not a heuristic — but it is a proxy: "angry" was IEMOCAP's
    label, not "frustrated", and this model was never validated against a
    frustration-specific ground truth.
  - `uncertainty` = normalized entropy of the 4-way softmax. When the model
    itself can't confidently pick a class, that ambiguity is used as a weak
    proxy for hesitant/uncertain delivery. This is honestly a stretch: this
    model was never trained to detect uncertainty, and entropy also rises
    for reasons that have nothing to do with hesitation (accent mismatch,
    noise). Treat this signal as the weaker of the two.
  - `confidence` = the classifier's own top-class probability (max of the
    softmax), i.e. "how sure the model is", not a statement about how sure
    the *speaker* sounded.
  - `energy` is not produced by this model at all — see acoustic/features.py.

Earliest point this output should be trusted: this model was fine-tuned on
IEMOCAP clips of several seconds; at this system's WINDOW_SIZE_MS=1500ms the
window is shorter than its typical training distribution, which is a real
(not fabricated) accuracy caveat — see README Q5 "stability across adjacent
windows" for the empirical follow-up.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .model import ModelUnavailableError
from .types import ModelInfo, RawModelOutput

__all__ = ["Wav2Vec2EmotionModel", "ModelUnavailableError"]

_MODEL_ID = "superb/wav2vec2-base-superb-er"
_VERSION = "wav2vec2-base-superb-er-r1"
_EXPECTED_SAMPLE_RATE = 16000


class Wav2Vec2EmotionModel:
    """Real, runnable adapter — lazy-loaded so importing this module (or
    constructing this class) never pulls torch/transformers into the
    default (mock-model) live path. Construction raises
    `ModelUnavailableError` for every failure mode (missing deps, missing
    weights, no network, corrupt cache) so callers can treat "real model
    requested but unusable" uniformly with "GPU/model server unavailable"
    (see README "Failure isolation" / PHASE 11)."""

    def __init__(self, model_id: str = _MODEL_ID, device: str = "cpu", cache_dir: str | None = None) -> None:
        self._model_id = model_id
        self._device = device
        self._cache_dir = cache_dir
        self._lock = threading.Lock()
        self._feature_extractor = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._info = ModelInfo(
            name=model_id,
            version=_VERSION,
            kind="real_wav2vec2_er",
            sample_rate=_EXPECTED_SAMPLE_RATE,
            min_duration_ms=1000.0,
            device=device,
            is_simulated=False,
            description=(
                "Pretrained wav2vec2-base encoder fine-tuned on IEMOCAP for "
                "4-class categorical emotion (neu/hap/ang/sad). frustration := "
                "P(angry); uncertainty := prediction entropy (weak proxy). "
                "See module docstring for full caveats."
            ),
        )
        self._load()  # eager: fail fast at construction, not on first predict()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise ModelUnavailableError(
                "torch/transformers not installed — install requirements-real-model.txt "
                "to enable the real acoustic model. Falling back is the caller's "
                "responsibility (see acoustic/factory.py)."
            ) from exc

        try:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._model_id, cache_dir=self._cache_dir, token=False,
            )
            model = AutoModelForAudioClassification.from_pretrained(
                self._model_id, cache_dir=self._cache_dir, token=False,
            )
            model.eval()
            model.to(self._device)
            self._model = model
            self._id2label = dict(model.config.id2label)
            self._torch = torch
        except Exception as exc:  # noqa: BLE001 - any failure here means "unavailable", not a crash
            raise ModelUnavailableError(f"failed to load {self._model_id}: {exc!r}") from exc

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, audio: np.ndarray, sample_rate: int) -> RawModelOutput:
        if audio.size == 0:
            raise ValueError("cannot predict on empty audio window")
        if sample_rate != _EXPECTED_SAMPLE_RATE:
            audio = _resample_linear(audio, sample_rate, _EXPECTED_SAMPLE_RATE)

        torch = self._torch
        inputs = self._feature_extractor(audio, sampling_rate=_EXPECTED_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**{k: v.to(self._device) for k, v in inputs.items()})
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().numpy()

        labels = [self._id2label[i] for i in range(len(probs))]
        prob_by_label: dict[str, float] = dict(zip(labels, probs.tolist()))
        frustration = float(prob_by_label.get("ang", 0.0))
        confidence = float(np.max(probs))

        eps = 1e-9
        entropy = float(-np.sum(probs * np.log(probs + eps)))
        max_entropy = float(np.log(len(probs)))
        uncertainty = float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy > 0 else 0.0

        return RawModelOutput(
            frustration=frustration,
            uncertainty=uncertainty,
            confidence=confidence,
            extra={"classProbabilities": prob_by_label},
        )


def _resample_linear(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Minimal dependency-free linear-interpolation resampler — adequate for
    this adapter's own use (this system's audio is always captured at
    16kHz already; this only guards against being handed something else,
    e.g. in a standalone benchmark script)."""
    if from_rate == to_rate or audio.size == 0:
        return audio
    duration_s = audio.size / from_rate
    n_out = int(round(duration_s * to_rate))
    x_old = np.linspace(0.0, duration_s, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, duration_s, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
