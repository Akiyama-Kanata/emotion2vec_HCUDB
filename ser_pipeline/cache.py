"""Fairseq-free sharded feature cache writer, validator, and mmap reader."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .audio import sha256_file
from .contracts import CACHE_SCHEMA_VERSION, FEATURE_LAYER, LABEL_ORDER
from .manifest import canonical_json, load_manifest, manifest_sha256, validate_manifest_records


@dataclass(frozen=True)
class CacheIndexEntry:
    dataset: str
    split: str
    utterance_id: str
    shard: str
    offset: int
    num_frames: int
    feature_dim: int
    class_index: int


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _write_index(entries: Iterable[CacheIndexEntry], path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as destination:
        for entry in entries:
            destination.write(canonical_json(asdict(entry)) + "\n")
    partial.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return payload


def load_index(path: str | Path) -> list[CacheIndexEntry]:
    result: list[CacheIndexEntry] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                result.append(CacheIndexEntry(**payload))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid cache index line {line_number}: {path}") from exc
    if not result:
        raise ValueError(f"cache index is empty: {path}")
    return result


def _cleanup_partials(directory: Path) -> list[str]:
    removed: list[str] = []
    if not directory.exists():
        return removed
    for current, _subdirs, names in os.walk(directory):
        for name in names:
            if not name.endswith(".partial"):
                continue
            path = Path(current) / name
            path.unlink()
            removed.append(str(path))
    return removed


def _validate_shard_meta(split_dir: Path, meta: Mapping[str, Any]) -> list[CacheIndexEntry]:
    shard_path = split_dir / str(meta.get("shard"))
    index_path = split_dir / str(meta.get("index"))
    if not shard_path.is_file() or not index_path.is_file():
        raise ValueError(f"cache shard files are missing in {split_dir}")
    if sha256_file(shard_path) != meta.get("shard_sha256"):
        raise ValueError(f"cache shard hash mismatch: {shard_path}")
    if sha256_file(index_path) != meta.get("index_sha256"):
        raise ValueError(f"cache index hash mismatch: {index_path}")
    array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"cache shard must be a non-empty 2D array: {shard_path}")
    if array.dtype != np.float32:
        raise ValueError(f"cache shard dtype must be float32: {shard_path}")
    if not np.isfinite(array).all():
        raise ValueError(f"cache shard contains non-finite values: {shard_path}")
    entries = load_index(index_path)
    expected_offset = 0
    for entry in entries:
        if entry.shard != shard_path.name:
            raise ValueError(f"cache index shard name mismatch: {index_path}")
        if entry.offset != expected_offset or entry.num_frames <= 0:
            raise ValueError(f"cache index offsets are invalid: {index_path}")
        if entry.feature_dim != array.shape[1]:
            raise ValueError(f"cache feature dimension mismatch: {index_path}")
        expected_offset += entry.num_frames
    if expected_offset != array.shape[0]:
        raise ValueError(f"cache index frame total mismatch: {index_path}")
    return entries


def completed_shards(split_dir: str | Path) -> tuple[list[dict[str, Any]], list[CacheIndexEntry]]:
    directory = Path(split_dir)
    metas: list[dict[str, Any]] = []
    entries: list[CacheIndexEntry] = []
    for meta_path in sorted(directory.glob("shard-*.meta.json")):
        meta = _load_json(meta_path)
        shard_entries = _validate_shard_meta(directory, meta)
        metas.append(meta)
        entries.extend(shard_entries)
    return metas, entries


def validate_success(split_dir: str | Path) -> dict[str, Any]:
    directory = Path(split_dir)
    success_path = directory / "_SUCCESS"
    if not success_path.is_file():
        raise ValueError(f"cache split is incomplete: {directory}")
    success = _load_json(success_path)
    if success.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"cache schema mismatch: {directory}")
    metas, entries = completed_shards(directory)
    if success.get("shards") != metas:
        raise ValueError(f"_SUCCESS shard metadata mismatch: {directory}")
    if int(success.get("utterance_count", -1)) != len(entries):
        raise ValueError(f"_SUCCESS utterance count mismatch: {directory}")
    return {"success": success, "entries": entries}


def cache_signature(meta: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "cache_schema_version",
        "encoder_name",
        "encoder_checkpoint_sha256",
        "feature_layer",
        "feature_dim",
        "dtype",
        "extraction_code_version",
        "manifest_sha256",
        "exclusion_contract",
        "duplicate_audit",
        "duplicate_exclusion_contract",
        "mapping_versions",
        "split_versions",
        "audio_preprocessing",
        "shard_policy",
    )
    return {field: meta.get(field) for field in fields}


def validate_cache(
    cache_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(cache_root)
    meta = _load_json(root / "cache_meta.json")
    if meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache_schema_version mismatch")
    if meta.get("feature_layer") != FEATURE_LAYER:
        raise ValueError("feature layer mismatch")
    if meta.get("dtype") != "float32":
        raise ValueError("cache dtype mismatch")
    actual_manifest_hash = manifest_sha256(manifest_path)
    if meta.get("manifest_sha256") != actual_manifest_hash:
        raise ValueError("cache manifest hash mismatch")
    if expected_signature is not None:
        actual = cache_signature(meta)
        for key, value in expected_signature.items():
            if actual.get(key) != value:
                raise ValueError(f"cache metadata mismatch for {key}")

    all_manifest_rows = load_manifest(manifest_path)
    manifest_validation = validate_manifest_records(all_manifest_rows)
    if meta.get("exclusion_contract") != manifest_validation["exclusion_contract"]:
        raise ValueError("cache exclusion contract provenance mismatch")
    if meta.get("duplicate_audit") != manifest_validation["duplicate_audit"]:
        raise ValueError("cache duplicate audit provenance mismatch")
    if meta.get("duplicate_exclusion_contract") != manifest_validation["duplicate_exclusion_contract"]:
        raise ValueError("cache duplicate exclusion contract provenance mismatch")
    manifest_rows = [row for row in all_manifest_rows if row["included"]]
    expected = {(row["dataset"], row["split"], row["utterance_id"]): row for row in manifest_rows}
    observed: dict[tuple[str, str, str], CacheIndexEntry] = {}
    split_reports: dict[str, Any] = {}
    for dataset, split in sorted({(row["dataset"], row["split"]) for row in manifest_rows}):
        directory = root / dataset / split
        validated = validate_success(directory)
        entries = validated["entries"]
        for entry in entries:
            key = (entry.dataset, entry.split, entry.utterance_id)
            if key in observed:
                raise ValueError(f"duplicate cached utterance: {key}")
            if key not in expected:
                raise ValueError(f"cached utterance is not included in manifest: {key}")
            if entry.class_index != int(expected[key]["class_index"]):
                raise ValueError(f"cached class_index mismatch: {key}")
            if entry.feature_dim != int(meta["feature_dim"]):
                raise ValueError(f"cached feature dimension mismatch: {key}")
            observed[key] = entry
        split_reports[f"{dataset}/{split}"] = {
            "utterances": len(entries),
            "shards": len(validated["success"]["shards"]),
        }
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"included manifest utterances are missing from cache: {missing[:5]}")
    if not bool(meta.get("complete")):
        raise ValueError("cache metadata is not marked complete")
    return {
        "status": "ok",
        "cache_id": meta.get("cache_id"),
        "manifest_sha256": actual_manifest_hash,
        "exclusion_contract": manifest_validation["exclusion_contract"],
        "duplicate_audit": manifest_validation["duplicate_audit"],
        "duplicate_exclusion_contract": manifest_validation["duplicate_exclusion_contract"],
        "utterances": len(observed),
        "feature_dim": int(meta["feature_dim"]),
        "splits": split_reports,
    }


class ShardedFeatureStore:
    """Read-only, lazily mmap-backed utterance feature lookup."""

    def __init__(self, cache_root: str | Path, manifest_path: str | Path, *, validate: bool = True):
        self.cache_root = Path(cache_root)
        self.manifest_path = Path(manifest_path)
        if validate:
            validate_cache(self.cache_root, self.manifest_path)
        self.meta = _load_json(self.cache_root / "cache_meta.json")
        self.records = {row["utterance_id"]: row for row in load_manifest(self.manifest_path) if row["included"]}
        self.entries: dict[str, CacheIndexEntry] = {}
        for dataset, split in sorted({(row["dataset"], row["split"]) for row in self.records.values()}):
            validated = validate_success(self.cache_root / dataset / split)
            for entry in validated["entries"]:
                if entry.utterance_id in self.entries:
                    raise ValueError(f"duplicate cache utterance_id: {entry.utterance_id}")
                self.entries[entry.utterance_id] = entry
        self._arrays: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def utterance_ids(self, *, dataset: str | None = None, split: str | None = None) -> list[str]:
        return [
            utterance_id
            for utterance_id, row in self.records.items()
            if (dataset is None or row["dataset"] == dataset) and (split is None or row["split"] == split)
        ]

    def get(self, utterance_id: str) -> np.ndarray:
        try:
            entry = self.entries[utterance_id]
        except KeyError as exc:
            raise KeyError(f"utterance is not in cache: {utterance_id}") from exc
        path = self.cache_root / entry.dataset / entry.split / entry.shard
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        array = self._arrays[path]
        return array[entry.offset : entry.offset + entry.num_frames]


FeatureCache = ShardedFeatureStore


__all__ = [
    "CacheIndexEntry",
    "FeatureCache",
    "ShardedFeatureStore",
    "cache_signature",
    "completed_shards",
    "load_index",
    "validate_cache",
    "validate_success",
    "_atomic_json",
    "_cleanup_partials",
    "_write_index",
]
