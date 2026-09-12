"""Dataset-independent SER feature, decoder, and official-head pipeline."""

from .contracts import (
    CACHE_SCHEMA_VERSION,
    CLASS_TO_INDEX,
    FEATURE_LAYER,
    LABEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    OFFICIAL_TARGET_ORDER,
    map_emotion,
)
from .diagnostics import OfficialTrainingDiagnosticsConfig

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CLASS_TO_INDEX",
    "FEATURE_LAYER",
    "LABEL_ORDER",
    "MANIFEST_SCHEMA_VERSION",
    "OFFICIAL_TARGET_ORDER",
    "OfficialTrainingDiagnosticsConfig",
    "map_emotion",
]
