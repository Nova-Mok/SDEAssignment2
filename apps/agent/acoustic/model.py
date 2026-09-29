"""The model interface every acoustic model implementation satisfies —
`acoustic/mock_model.py` (the default, DSP-based) and
`acoustic/real_model.py` (the pretrained wav2vec2 adapter). Nothing in the
pipeline imports either implementation directly; it only ever depends on
this Protocol, which is what makes swapping models a one-line config change
(`ACOUSTIC_MODEL_KIND=real`) rather than a code change.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import ModelInfo, RawModelOutput

if TYPE_CHECKING:
    import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be constructed or used at all — missing
    weights, missing optional dependency (torch/transformers), no network on
    first download, GPU/model server unreachable. The pipeline treats this
    as "acoustic intelligence is off for this call", never as a reason to
    fail the call itself. See README "Failure isolation"."""


@runtime_checkable
class AcousticModel(Protocol):
    @property
    def info(self) -> ModelInfo:
        """Static model metadata. Must not raise; construct eagerly enough
        that this is always answerable once __init__ succeeds."""
        ...

    def predict(self, audio: "np.ndarray", sample_rate: int) -> RawModelOutput:
        """Synchronous, CPU/GPU-bound prediction over one window of mono
        float32 audio in [-1, 1]. Called from `asyncio.to_thread` by the
        pipeline — implementations should NOT spawn their own threads or
        assume an event loop is available.

        Must raise (not return a degraded/garbage value) if it cannot
        produce a real prediction — the pipeline's failure isolation
        depends on exceptions being the only failure signal.
        """
        ...
