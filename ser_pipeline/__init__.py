"""Dataset-independent four-class SER feature/decoder pipeline."""

from .contracts import (
    CACHE_SCHEMA_VERSION,
    CLASS_TO_INDEX,
    FEATURE_LAYER,
    LABEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    map_emotion,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CLASS_TO_INDEX",
    "FEATURE_LAYER",
    "LABEL_ORDER",
    "MANIFEST_SCHEMA_VERSION",
    "map_emotion",
]
