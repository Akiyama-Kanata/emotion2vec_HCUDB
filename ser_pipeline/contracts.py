"""Versioned constants and label mapping contracts for the SER study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "ser_manifest_v1"
CACHE_SCHEMA_VERSION = "ser_feature_cache_v1"
CHECKPOINT_SCHEMA_VERSION = "ser_decoder_checkpoint_v1"
RESULT_SCHEMA_VERSION = "ser_evaluation_result_v1"
FEATURE_LAYER = "final_after_encoder_norm"
EXTRACTION_CODE_VERSION = "ser_features_v1"
LABEL_ORDER = ("anger", "happy", "sadness", "disgust")
CLASS_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
SUPPORTED_DATASETS = ("msp_podcast", "hcudb1", "iemocap")
EXPECTED_INCLUDED_COUNTS = {"msp_podcast": 25111, "hcudb1": 2100, "iemocap": 3825}
RESULT_LIMITATIONS = (
    {
        "id": "emotion2vec_pretraining_includes_msp_podcast_v1_8",
        "status": "verified",
        "source": "https://aclanthology.org/2024.findings-acl.931/",
        "implication": "MSP-Podcast evaluation is not fully unseen with respect to encoder pre-training data.",
    },
    {
        "id": "msp_podcast_v1_8_is_complete_subset_of_r1_10",
        "status": "unverified",
        "reason": "Release 1.8 metadata is not locally available.",
    },
)

MANIFEST_FIELDS = (
    "manifest_schema_version",
    "dataset",
    "dataset_release",
    "utterance_id",
    "audio_relpath",
    "audio_sha256",
    "speaker_id",
    "speaker_id_status",
    "group_id",
    "session_id",
    "source_split",
    "split",
    "split_version",
    "original_emotion",
    "mapped_emotion",
    "class_index",
    "mapping_version",
    "included",
    "exclusion_reasons",
    "approximate_mapping",
    "audio_size_bytes",
    "sample_rate_hz",
    "channels",
    "num_samples",
    "duration_seconds",
)

_CONFIG_PATH = Path(__file__).with_name("config") / "mappings.v1.json"


@lru_cache(maxsize=4)
def load_mapping_config(path: str | Path | None = None) -> dict[str, Any]:
    mapping_path = Path(path) if path is not None else _CONFIG_PATH
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if tuple(payload.get("label_order", ())) != LABEL_ORDER:
        raise ValueError(f"mapping label_order must be {list(LABEL_ORDER)}")
    if set(payload.get("datasets", {})) != set(SUPPORTED_DATASETS):
        raise ValueError("mapping config must define exactly the supported datasets")
    return payload


@dataclass(frozen=True)
class MappingDecision:
    dataset: str
    original_emotion: str
    mapped_emotion: str | None
    class_index: int | None
    mapping_version: str
    included: bool
    exclusion_reasons: tuple[str, ...]
    approximate_mapping: bool


def dataset_contract(dataset: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = str(dataset).strip().lower()
    payload = load_mapping_config() if config is None else config
    try:
        return payload["datasets"][normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset!r}") from exc


def map_emotion(
    dataset: str,
    original_emotion: str,
    *,
    config: dict[str, Any] | None = None,
) -> MappingDecision:
    normalized_dataset = str(dataset).strip().lower()
    label = str(original_emotion).strip()
    contract = dataset_contract(normalized_dataset, config)
    mappings = contract["mappings"]
    excluded = set(contract["excluded_labels"])
    known = set(mappings) | excluded
    if label not in known:
        raise ValueError(f"unknown {normalized_dataset} emotion label: {label!r}")
    mapped = mappings.get(label)
    included = mapped is not None
    return MappingDecision(
        dataset=normalized_dataset,
        original_emotion=label,
        mapped_emotion=mapped,
        class_index=CLASS_TO_INDEX[mapped] if mapped is not None else None,
        mapping_version=contract["mapping_version"],
        included=included,
        exclusion_reasons=() if included else ("label_not_in_primary_4",),
        approximate_mapping=label in set(contract.get("approximate_labels", ())),
    )


def normalize_layer(layer: str | int) -> str:
    """Accept only the explicitly supported final encoder representation."""
    if layer == "final":
        return FEATURE_LAYER
    raise ValueError("--layer supports only 'final'; integer/intermediate layers are not defined")
