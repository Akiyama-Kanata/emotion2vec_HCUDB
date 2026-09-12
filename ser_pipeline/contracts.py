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
OFFICIAL_CHECKPOINT_SCHEMA_VERSION = "ser_official_checkpoint_v2"
PRIMARY_EVALUATION_METHOD = "primary_logits_softmax_v1"
RESULT_SCHEMA_VERSION = "ser_evaluation_result_v1"
OFFICIAL_EVALUATION_METHOD = "official9_argmax_target6_metrics_v2"
OFFICIAL_RESULT_SCHEMA_VERSION = "ser_official_evaluation_result_v2"
FEATURE_LAYER = "final_after_encoder_norm"
EXTRACTION_CODE_VERSION = "ser_features_v1"
LABEL_ORDER = ("anger", "happy", "sadness", "disgust")
CLASS_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
LABEL_PROFILES = ("ab4", "official6")
OFFICIAL_TARGET_ORDER = ("angry", "disgusted", "fearful", "happy", "sad", "surprised")
OFFICIAL_TARGET_TO_INDEX = {label: index for index, label in enumerate(OFFICIAL_TARGET_ORDER)}
SUPPORTED_DATASETS = ("msp_podcast", "hcudb1", "iemocap")
EXPECTED_INCLUDED_COUNTS = {"msp_podcast": 24857, "hcudb1": 2100, "iemocap": 3825}
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
_OFFICIAL6_CONFIG_PATH = Path(__file__).with_name("config") / "mappings.official6.v1.json"


def label_order_for_profile(label_profile: str) -> tuple[str, ...]:
    normalized = str(label_profile).strip().lower()
    if normalized == "ab4":
        return LABEL_ORDER
    if normalized == "official6":
        return OFFICIAL_TARGET_ORDER
    raise ValueError(f"unsupported label profile: {label_profile!r}")


@lru_cache(maxsize=8)
def load_mapping_config(
    path: str | Path | None = None,
    *,
    label_profile: str = "ab4",
) -> dict[str, Any]:
    normalized_profile = str(label_profile).strip().lower()
    label_order = label_order_for_profile(normalized_profile)
    mapping_path = Path(path) if path is not None else (
        _CONFIG_PATH if normalized_profile == "ab4" else _OFFICIAL6_CONFIG_PATH
    )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if payload.get("label_profile", "ab4") != normalized_profile:
        raise ValueError("mapping label_profile mismatch")
    if tuple(payload.get("label_order", ())) != label_order:
        raise ValueError(f"mapping label_order must be {list(label_order)}")
    expected_datasets = set(SUPPORTED_DATASETS if normalized_profile == "ab4" else ("msp_podcast", "hcudb1"))
    if set(payload.get("datasets", {})) != expected_datasets:
        raise ValueError("mapping config does not define exactly the datasets for its label profile")
    versions = [str(contract.get("mapping_version", "")) for contract in payload["datasets"].values()]
    if any(not value for value in versions) or len(versions) != len(set(versions)):
        raise ValueError("mapping versions must be non-empty and unique within a profile")
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


def dataset_contract(
    dataset: str,
    config: dict[str, Any] | None = None,
    *,
    label_profile: str = "ab4",
) -> dict[str, Any]:
    normalized = str(dataset).strip().lower()
    payload = load_mapping_config(label_profile=label_profile) if config is None else config
    try:
        return payload["datasets"][normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset!r}") from exc


def map_emotion(
    dataset: str,
    original_emotion: str,
    *,
    config: dict[str, Any] | None = None,
    label_profile: str = "ab4",
) -> MappingDecision:
    normalized_dataset = str(dataset).strip().lower()
    label = str(original_emotion).strip()
    contract = dataset_contract(normalized_dataset, config, label_profile=label_profile)
    label_order = label_order_for_profile(label_profile)
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
        class_index=label_order.index(mapped) if mapped is not None else None,
        mapping_version=contract["mapping_version"],
        included=included,
        exclusion_reasons=() if included else (
            "label_not_in_primary_4" if label_profile == "ab4" else "label_not_in_official6",
        ),
        approximate_mapping=label in set(contract.get("approximate_labels", ())),
    )


def label_profile_for_mapping_version(mapping_version: str) -> str:
    """Resolve a manifest mapping version to exactly one label profile."""
    matches = []
    for profile in LABEL_PROFILES:
        config = load_mapping_config(label_profile=profile)
        if any(contract["mapping_version"] == mapping_version for contract in config["datasets"].values()):
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError(f"mapping_version does not resolve to one label profile: {mapping_version!r}")
    return matches[0]


def normalize_layer(layer: str | int) -> str:
    """Accept only the explicitly supported final encoder representation."""
    if layer == "final":
        return FEATURE_LAYER
    raise ValueError("--layer supports only 'final'; integer/intermediate layers are not defined")
