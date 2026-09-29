"""Turns an `AcousticConfig` into a constructed `AcousticModel` (or `None`),
isolating every model-construction failure mode — missing optional
dependency, missing/unreachable weights, disabled by config — into one
place. `None` always means "acoustic intelligence is off for this call";
nothing downstream needs to know *why*.
"""
from __future__ import annotations

import logging

from .config import AcousticConfig
from .model import AcousticModel, ModelUnavailableError

logger = logging.getLogger("acoustic.factory")


def build_acoustic_model(config: AcousticConfig) -> AcousticModel | None:
    if not config.ENABLED:
        return None

    if config.MODEL_KIND == "mock":
        from .mock_model import DeterministicProsodyModel

        return DeterministicProsodyModel()

    if config.MODEL_KIND == "real":
        from .real_model import Wav2Vec2EmotionModel

        try:
            return Wav2Vec2EmotionModel()
        except ModelUnavailableError:
            logger.warning(
                "real acoustic model unavailable — acoustic intelligence disabled for this "
                "call (call continues normally; see README 'Failure isolation')",
                exc_info=True,
            )
            return None

    logger.error("unknown ACOUSTIC_MODEL_KIND=%r — acoustic intelligence disabled", config.MODEL_KIND)
    return None
