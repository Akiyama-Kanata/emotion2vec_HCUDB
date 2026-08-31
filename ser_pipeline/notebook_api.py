"""Small orchestration helpers used by the two study notebooks."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .audio import sha256_file
from .cache import CacheIndexEntry, _atomic_json, _write_index, validate_cache
from .contracts import (
    CACHE_SCHEMA_VERSION,
    EXTRACTION_CODE_VERSION,
    FEATURE_LAYER,
    LABEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    load_mapping_config,
)
from .manifest import manifest_sha256, write_manifest
from .splits import IEMOCAP_SPLIT_VERSION, MSP_SPLIT_VERSION, load_hcudb_split
from .study import DatasetArtifacts, EVALUATION_DATASETS, STUDY_SEEDS, run_transfer_study
from .training import TrainingConfig


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "label_order": list(LABEL_ORDER),
        "feature_layer": FEATURE_LAYER,
    }


def mapping_summary() -> list[dict[str, Any]]:
    config = load_mapping_config()
    rows = []
    for dataset, contract in config["datasets"].items():
        for source, mapped in contract["mappings"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "mapping_version": contract["mapping_version"],
                    "original_emotion": source,
                    "mapped_emotion": mapped,
                    "included": True,
                    "approximate": source in contract.get("approximate_labels", []),
                }
            )
        for source in contract["excluded_labels"]:
            rows.append(
                {
                    "dataset": dataset,
                    "mapping_version": contract["mapping_version"],
                    "original_emotion": source,
                    "mapped_emotion": None,
                    "included": False,
                    "approximate": False,
                }
            )
    return rows


def split_summary() -> dict[str, Any]:
    return {
        "msp_podcast": {
            "version": MSP_SPLIT_VERSION,
            "assignment": {"Train": "train", "Development": "validation", "Test1": "test", "Test2": "excluded"},
        },
        "hcudb1": load_hcudb_split(),
        "iemocap": {"version": IEMOCAP_SPLIT_VERSION, "assignment": "all_sessions -> test"},
    }


def extraction_command_preview(
    manifest: str = "<manifest.jsonl>",
    audio_root: str = "<dataset-root>",
    cache_root: str = "<cache-root>",
) -> str:
    return (
        "python -m ser_pipeline extract-features "
        f"--manifest {manifest} --audio-root {audio_root} --cache-root {cache_root} "
        "--user-dir <upstream-dir> --checkpoint <base-checkpoint> --layer final --device auto"
    )


def one_item_feature_benchmark(feature_dim: int = 768, seconds: float = 1.0) -> dict[str, Any]:
    start = time.perf_counter()
    frames = max(1, int(seconds * 50))
    features = np.zeros((frames, feature_dim), dtype=np.float32)
    elapsed = time.perf_counter() - start
    return {
        "mode": "synthetic_preflight_only",
        "duration_seconds": float(seconds),
        "feature_frames": frames,
        "feature_dim": feature_dim,
        "feature_bytes": int(features.nbytes),
        "elapsed_seconds": float(elapsed),
    }


def _record(dataset: str, split: str, class_index: int, serial: int) -> dict[str, Any]:
    label = LABEL_ORDER[class_index]
    utterance_id = f"{dataset}_{split}_{class_index}_{serial}"
    audio_hash = hashlib.sha256(utterance_id.encode("utf-8")).hexdigest()
    mapping = {
        "msp_podcast": ("R1.10", "msp_podcast_r1_10_primary_v1", MSP_SPLIT_VERSION),
        "hcudb1": ("HCUDB1", "hcudb1_acted_emotion_v1", "hcudb1_speaker_split_v1"),
        "iemocap": ("IEMOCAP_full_release", "iemocap_external_v1", IEMOCAP_SPLIT_VERSION),
    }[dataset]
    originals = {
        "msp_podcast": ("A", "H", "S", "D"),
        "hcudb1": ("怒り", "狂喜・楽しい", "憂鬱・悲しい", "嫌い"),
        "iemocap": ("ang", "hap", "sad", "dis"),
    }[dataset]
    if dataset == "msp_podcast":
        source_split = {"train": "Train", "validation": "Development", "test": "Test1"}[split]
        speaker = f"msp_{split}_{serial}"
        session = "podcast"
    elif dataset == "hcudb1":
        source_split = "all"
        speaker = {"train": "FA", "validation": "FF", "test": "FG"}[split]
        session = ""
    else:
        source_split = "all_sessions"
        speaker = f"Ses0{serial % 5 + 1}F"
        session = f"Ses0{serial % 5 + 1}"
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_release": mapping[0],
        "utterance_id": utterance_id,
        "audio_relpath": f"unavailable/{utterance_id}.wav",
        "audio_sha256": audio_hash,
        "speaker_id": speaker,
        "speaker_id_status": "known",
        "group_id": speaker,
        "session_id": session,
        "source_split": source_split,
        "split": split,
        "split_version": mapping[2],
        "original_emotion": originals[class_index],
        "mapped_emotion": label,
        "class_index": class_index,
        "mapping_version": mapping[1],
        "included": True,
        "exclusion_reasons": [],
        "approximate_mapping": dataset == "hcudb1" and class_index == 3,
        "audio_size_bytes": 100,
        "sample_rate_hz": 16000,
        "channels": 1,
        "num_samples": 800,
        "duration_seconds": 0.05,
    }


def _demo_rows(dataset: str) -> list[dict[str, Any]]:
    if dataset == "iemocap":
        return [_record(dataset, "test", class_index, serial) for serial in range(2) for class_index in range(4)]
    rows = []
    for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
        for serial in range(repeats):
            for class_index in range(4):
                rows.append(_record(dataset, split, class_index, serial))
    return rows


def _demo_features(row: dict[str, Any], feature_dim: int) -> np.ndarray:
    frames = 3 + int(hashlib.sha256(row["utterance_id"].encode()).digest()[0] % 3)
    vector = np.zeros(feature_dim, dtype=np.float32)
    vector[int(row["class_index"])] = 3.0
    vector[4:] = (int(row["class_index"]) + 1) / 10.0
    return np.repeat(vector[None, :], frames, axis=0)


def _write_demo_cache(cache_root: Path, manifest_path: Path, rows: list[dict[str, Any]], feature_dim: int) -> None:
    encoder_hash = "d" * 64
    for split in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == split]
        directory = cache_root / rows[0]["dataset"] / split
        directory.mkdir(parents=True, exist_ok=True)
        arrays = [_demo_features(row, feature_dim) for row in subset]
        concatenated = np.concatenate(arrays, axis=0).astype(np.float32)
        shard_name = "shard-00000.npy"
        index_name = "shard-00000.index.jsonl"
        shard_path = directory / shard_name
        np.save(shard_path, concatenated, allow_pickle=False)
        entries = []
        offset = 0
        for row, array in zip(subset, arrays):
            entries.append(
                CacheIndexEntry(
                    dataset=row["dataset"],
                    split=split,
                    utterance_id=row["utterance_id"],
                    shard=shard_name,
                    offset=offset,
                    num_frames=len(array),
                    feature_dim=feature_dim,
                    class_index=row["class_index"],
                )
            )
            offset += len(array)
        index_path = directory / index_name
        _write_index(entries, index_path)
        shard_meta = {
            "shard": shard_name,
            "index": index_name,
            "shard_sha256": sha256_file(shard_path),
            "index_sha256": sha256_file(index_path),
            "frames": int(concatenated.shape[0]),
            "utterances": len(entries),
            "feature_dim": feature_dim,
            "dtype": "float32",
        }
        _atomic_json(shard_meta, directory / "shard-00000.meta.json")
        _atomic_json(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "dataset": rows[0]["dataset"],
                "split": split,
                "utterance_count": len(entries),
                "shards": [shard_meta],
            },
            directory / "_SUCCESS",
        )
    mapping_versions = sorted({row["mapping_version"] for row in rows})
    split_versions = sorted({row["split_version"] for row in rows})
    cache_id = hashlib.sha256((rows[0]["dataset"] + str(feature_dim)).encode()).hexdigest()[:16]
    _atomic_json(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_id": cache_id,
            "encoder_name": "synthetic_demo_encoder",
            "encoder_checkpoint_sha256": encoder_hash,
            "feature_layer": FEATURE_LAYER,
            "feature_dim": feature_dim,
            "dtype": "float32",
            "extraction_code_version": EXTRACTION_CODE_VERSION,
            "git_commit": "demo",
            "manifest_sha256": manifest_sha256(manifest_path),
            "mapping_versions": mapping_versions,
            "split_versions": split_versions,
            "audio_preprocessing": {
                "target_sample_rate_hz": 16000,
                "channels": "mono_required",
                "resampler": "scipy.signal.resample_poly",
            },
            "shard_policy": {"max_frames_approximately": 65536},
            "complete": True,
        },
        cache_root / "cache_meta.json",
    )


def make_demo_artifacts(
    root: str | Path,
    feature_dim: int = 8,
    *,
    datasets: Sequence[str] = ("msp_podcast", "hcudb1", "iemocap"),
) -> dict[str, DatasetArtifacts]:
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for dataset in datasets:
        rows = _demo_rows(dataset)
        manifest_path = destination / dataset / "manifest.jsonl"
        write_manifest(rows, manifest_path)
        cache_root = destination / dataset / "cache"
        _write_demo_cache(cache_root, manifest_path, rows, feature_dim)
        validate_cache(cache_root, manifest_path)
        artifacts[dataset] = DatasetArtifacts(manifest_path=manifest_path, cache_root=cache_root)
    return artifacts


def demo_cache_summary(
    root: str | Path,
    *,
    datasets: Sequence[str] = ("msp_podcast", "hcudb1", "iemocap"),
) -> dict[str, Any]:
    artifacts = make_demo_artifacts(root, datasets=datasets)
    return {
        dataset: validate_cache(current.cache_root, current.manifest_path)
        for dataset, current in artifacts.items()
    }


def run_demo_transfer_study(
    root: str | Path,
    *,
    seeds=STUDY_SEEDS,
    epochs: int = 1,
) -> dict[str, Any]:
    destination = Path(root)
    artifacts = make_demo_artifacts(destination / "artifacts", datasets=EVALUATION_DATASETS)
    config = TrainingConfig(
        seed=int(seeds[0]),
        device="cpu",
        epochs=epochs,
        batch_size=4,
        learning_rate=0.01,
        hidden_dim=8,
        dropout=0.0,
    )
    return run_transfer_study(artifacts, destination / "study", seeds=seeds, base_config=config)


__all__ = [
    "STUDY_SEEDS",
    "demo_cache_summary",
    "environment_summary",
    "extraction_command_preview",
    "make_demo_artifacts",
    "mapping_summary",
    "one_item_feature_benchmark",
    "run_demo_transfer_study",
    "split_summary",
]
