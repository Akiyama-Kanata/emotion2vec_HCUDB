"""Fairseq-free sharded feature cache writer, validator, and mmap reader."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from .audio import sha256_file
from .contracts import CACHE_SCHEMA_VERSION, FEATURE_LAYER
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
    try:
        array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid cache shard array: {shard_path}") from exc
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
    if int(meta.get("frames", -1)) != int(array.shape[0]):
        raise ValueError(f"cache shard metadata frame count mismatch: {shard_path}")
    if int(meta.get("utterances", -1)) != len(entries):
        raise ValueError(f"cache shard metadata utterance count mismatch: {shard_path}")
    if int(meta.get("feature_dim", -1)) != int(array.shape[1]):
        raise ValueError(f"cache shard metadata feature dimension mismatch: {shard_path}")
    if meta.get("dtype") != "float32":
        raise ValueError(f"cache shard metadata dtype mismatch: {shard_path}")
    return entries


def completed_shards(split_dir: str | Path) -> tuple[list[dict[str, Any]], list[CacheIndexEntry]]:
    directory = Path(split_dir)
    metas: list[dict[str, Any]] = []
    entries: list[CacheIndexEntry] = []
    meta_paths = sorted(directory.glob("shard-*.meta.json"))
    for shard_number, meta_path in enumerate(meta_paths):
        expected_stem = f"shard-{shard_number:05d}"
        if meta_path.name != f"{expected_stem}.meta.json":
            raise ValueError(f"cache shard numbers must be contiguous from zero: {meta_path}")
        meta = _load_json(meta_path)
        if meta.get("shard") != f"{expected_stem}.npy" or meta.get("index") != f"{expected_stem}.index.jsonl":
            raise ValueError(f"cache shard metadata names do not match commit marker: {meta_path}")
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


def validate_official_cache(meta, snapshot=None):
    """Reject Base caches and incomplete or mismatched Large provenance."""
    if meta.get("encoder_name") != "emotion2vec_plus_large" or meta.get("feature_dim") != 1024:
        raise ValueError("official head requires a Large 1024-dimensional cache")
    if meta.get("dtype") != "float32" or meta.get("feature_layer") != FEATURE_LAYER:
        raise ValueError("official cache feature contract mismatch")
    provenance = meta.get("official_provenance") or {}
    identity = provenance.get("snapshot") or {}
    required = {"revision", "checkpoint_sha256", "config_sha256", "tokens_sha256", "head_sha256", "label_spec", "normalize", "mask", "remove_extra_tokens"}
    if not required <= identity.keys() or not identity["revision"]:
        raise ValueError("official cache snapshot provenance is incomplete")
    for key in ("checkpoint_sha256", "config_sha256", "tokens_sha256", "head_sha256"):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid official cache {key}")
    if identity["mask"] is not False or identity["remove_extra_tokens"] is not True or type(identity["normalize"]) is not bool:
        raise ValueError("official extraction settings mismatch")
    if (provenance.get("extraction_code_version") != "ser_official_features_v1"
            or meta.get("extraction_code_version") != "ser_official_features_v1"
            or not provenance.get("dependencies") or not provenance.get("implementation_sha256")
            or set(provenance.get("implementation_files_sha256", {})) != {"audio.py", "features.py", "model.py", "official.py"}):
        raise ValueError("official extraction implementation provenance is incomplete")
    if meta.get("encoder_checkpoint_sha256") != identity["checkpoint_sha256"]:
        raise ValueError("official cache checkpoint hash mismatch")
    if snapshot is not None and identity != snapshot:
        raise ValueError("official cache snapshot mismatch")
    return provenance


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
        "official_provenance",
    )
    return {field: meta.get(field) for field in fields}


@dataclass
class _ValidatedCache:
    report: dict[str, Any]
    meta: dict[str, Any]
    records: dict[str, dict[str, Any]]
    entries: dict[str, CacheIndexEntry]


def _validate_cache(
    cache_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> _ValidatedCache:
    root = Path(cache_root)
    meta = _load_json(root / "cache_meta.json")
    if meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache_schema_version mismatch")
    if meta.get("feature_layer") != FEATURE_LAYER:
        raise ValueError("feature layer mismatch")
    if meta.get("dtype") != "float32":
        raise ValueError("cache dtype mismatch")
    if meta.get("encoder_name") == "emotion2vec_plus_large" or "official_provenance" in meta:
        validate_official_cache(meta)
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
            if (entry.dataset, entry.split) != (dataset, split):
                raise ValueError(f"cached dataset/split directory mismatch: {directory}")
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
    records_by_id = {row["utterance_id"]: row for row in manifest_rows}
    entries_by_id = {entry.utterance_id: entry for entry in observed.values()}
    if len(entries_by_id) != len(observed):
        raise ValueError("duplicate cache utterance_id")
    report = {
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
    return _ValidatedCache(report, meta, records_by_id, entries_by_id)


def validate_cache(
    cache_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fully validate a cache, retaining the public JSON-compatible report API."""
    return _validate_cache(cache_root, manifest_path, expected_signature=expected_signature).report


def _uncommitted_fragments(root: Path) -> list[Path]:
    """Return only files that are safe to discard because no shard meta commits them."""
    if not root.exists():
        return []
    partials = [path for path in root.rglob("*") if path.is_file() and path.name.endswith(".partial")]
    orphaned: list[Path] = []
    for pattern, suffix in (("shard-*.npy", ".npy"), ("shard-*.index.jsonl", ".index.jsonl")):
        for path in root.rglob(pattern):
            stem = path.name[: -len(suffix)]
            if not re.fullmatch(r"shard-\d{5}", stem):
                continue
            if not (path.parent / f"{stem}.meta.json").is_file():
                orphaned.append(path)
    return sorted(set(partials + orphaned), key=lambda path: str(path))


def inspect_cache_resume(
    cache_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
    expected_dim: int = 1024,
    max_shard_frames: int = 65536,
) -> dict[str, Any]:
    """Read-only audit of committed shards and safe-to-remove uncommitted files."""
    root = Path(cache_root)
    all_rows = load_manifest(manifest_path)
    validation = validate_manifest_records(all_rows)
    included = [row for row in all_rows if bool(row["included"])]
    if not included:
        raise ValueError("manifest has no included rows")
    actual_manifest_hash = manifest_sha256(manifest_path)
    expected_splits = {
        (str(row["dataset"]), str(row["split"]))
        for row in included
    }
    fragments = _uncommitted_fragments(root)
    meta_path = root / "cache_meta.json"
    committed = 0
    split_reports: dict[str, Any] = {}

    if not meta_path.is_file():
        committed_meta = list(root.glob("*/*/shard-*.meta.json")) if root.exists() else []
        success_files = list(root.glob("*/*/_SUCCESS")) if root.exists() else []
        if committed_meta or success_files:
            raise ValueError("cannot resume committed cache shards without cache_meta.json")
        return {
            "status": "new",
            "cache_root": str(root.resolve()),
            "manifest_sha256": actual_manifest_hash,
            "complete": False,
            "committed_utterances": 0,
            "pending_utterances": len(included),
            "recoverable_fragments": [str(path.resolve()) for path in fragments],
            "splits": {},
        }

    meta = _load_json(meta_path)
    if meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache_schema_version mismatch")
    if meta.get("manifest_sha256") != actual_manifest_hash:
        raise ValueError("cache manifest hash mismatch")
    if meta.get("feature_layer") != FEATURE_LAYER or meta.get("dtype") != "float32":
        raise ValueError("cache feature contract mismatch")
    if int(meta.get("feature_dim", -1)) != int(expected_dim):
        raise ValueError("cache feature dimension mismatch")
    if meta.get("shard_policy") != {"max_frames_approximately": int(max_shard_frames)}:
        raise ValueError("cache shard policy mismatch")
    if meta.get("encoder_name") == "emotion2vec_plus_large" or "official_provenance" in meta:
        validate_official_cache(meta)
    if expected_signature is not None:
        actual_signature = cache_signature(meta)
        for key, value in expected_signature.items():
            if actual_signature.get(key) != value:
                raise ValueError(f"cache metadata mismatch for {key}")
    for key in ("exclusion_contract", "duplicate_audit", "duplicate_exclusion_contract"):
        if meta.get(key) != validation[key]:
            raise ValueError(f"cache {key} provenance mismatch")

    observed_ids: set[str] = set()
    actual_committed_splits = {
        (path.parent.parent.name, path.parent.name)
        for path in root.glob("*/*/shard-*.meta.json")
    }
    actual_success_splits = {
        (path.parent.parent.name, path.parent.name)
        for path in root.glob("*/*/_SUCCESS")
    }
    unexpected = sorted((actual_committed_splits | actual_success_splits) - expected_splits)
    if unexpected:
        raise ValueError(f"cache contains unexpected dataset/split directories: {unexpected}")

    for dataset, split in sorted(expected_splits):
        split_rows = [
            row for row in included
            if str(row["dataset"]) == dataset and str(row["split"]) == split
        ]
        directory = root / dataset / split
        metas, entries = completed_shards(directory)
        expected_prefix = [str(row["utterance_id"]) for row in split_rows[: len(entries)]]
        observed = [entry.utterance_id for entry in entries]
        if observed != expected_prefix:
            raise ValueError(f"resume cache utterance prefix mismatch: {dataset}/{split}")
        for entry, row in zip(entries, split_rows):
            if (entry.dataset, entry.split) != (dataset, split):
                raise ValueError(f"cached dataset/split directory mismatch: {directory}")
            if entry.utterance_id in observed_ids:
                raise ValueError(f"duplicate cached utterance_id: {entry.utterance_id}")
            if entry.class_index != int(row["class_index"]):
                raise ValueError(f"cached class_index mismatch: {entry.utterance_id}")
            if entry.feature_dim != int(expected_dim):
                raise ValueError(f"cached feature dimension mismatch: {entry.utterance_id}")
            observed_ids.add(entry.utterance_id)
        success_path = directory / "_SUCCESS"
        split_complete = success_path.is_file()
        if split_complete:
            success = validate_success(directory)
            if len(entries) != len(split_rows):
                raise ValueError(f"completed cache utterance order mismatch: {dataset}/{split}")
            if success["entries"] != entries:
                raise ValueError(f"completed cache index changed during inspection: {dataset}/{split}")
        committed += len(entries)
        split_reports[f"{dataset}/{split}"] = {
            "complete": split_complete,
            "committed_utterances": len(entries),
            "pending_utterances": len(split_rows) - len(entries),
            "shards": len(metas),
        }

    pending = len(included) - committed
    complete = bool(meta.get("complete"))
    all_splits_complete = all(item["complete"] for item in split_reports.values())
    if complete and not (all_splits_complete and pending == 0):
        raise ValueError("cache complete flag is inconsistent with committed splits")
    return {
        "status": "complete" if complete else "resumable",
        "cache_root": str(root.resolve()),
        "manifest_sha256": actual_manifest_hash,
        "cache_id": meta.get("cache_id"),
        "complete": complete,
        "committed_utterances": committed,
        "pending_utterances": pending,
        "recoverable_fragments": [str(path.resolve()) for path in fragments],
        "splits": split_reports,
    }


def cleanup_uncommitted_cache_fragments(
    cache_root: str | Path,
    fragments: Iterable[str | Path],
) -> list[str]:
    """Delete only fragment paths previously returned by :func:`inspect_cache_resume`."""
    root = Path(cache_root).resolve()
    removed: list[str] = []
    for value in fragments:
        path = Path(value).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"cache fragment is outside cache root: {path}") from exc
        name = path.name
        safe = name.endswith(".partial") or bool(
            re.fullmatch(r"shard-\d{5}(?:\.npy|\.index\.jsonl)", name)
        )
        if not safe:
            raise ValueError(f"refusing to remove non-fragment cache file: {path}")
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


class ShardedFeatureStore:
    """Process-local validated index and lazy, read-only mmap feature lookup.

    Call ensure_validated before reusing the store in another operation. File
    identity, size and timestamps detect ordinary changes; this is not a lock
    against concurrent writers or metadata-preserving tampering. Cache inputs
    must remain read-only during a training/evaluation operation.
    """

    def __init__(self, cache_root: str | Path, manifest_path: str | Path, *, validate: bool = True):
        # Retain the old keyword for callers, but never use False as proof that
        # a cache was validated. Reuse this object to avoid a second full pass.
        self.cache_root = Path(cache_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.meta: dict[str, Any] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.entries: dict[str, CacheIndexEntry] = {}
        self._arrays: dict[Path, np.ndarray] = {}
        self._snapshot = None
        self._validation_pid: int | None = None
        self.validation_report: dict[str, Any] = {}
        self.validation_seconds = 0.0
        self.validation_count = 0
        self.ensure_validated()

    def _input_snapshot(self) -> dict[Path, tuple]:
        paths = {self.manifest_path, self.cache_root / "cache_meta.json"}
        paths.update(self.cache_root.glob("*/*/_SUCCESS"))
        paths.update(self.cache_root.glob("*/*/shard-*"))
        # Follow referenced payload names as well as the writer's usual names.
        # Only these small JSON files are read during an unchanged reuse check.
        for meta_path in self.cache_root.glob("*/*/shard-*.meta.json"):
            meta = _load_json(meta_path)
            paths.update(meta_path.parent / str(meta.get(key)) for key in ("shard", "index"))
        snapshot = {}
        for path in paths:
            try:
                stat = path.stat()
            except OSError as exc:
                raise ValueError(f"cache input is missing or inaccessible: {path}") from exc
            snapshot[path] = (
                str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns,
            )
        return snapshot

    def ensure_validated(self) -> dict[str, Any]:
        """Reuse an unchanged validation, otherwise discard old maps and revalidate."""
        try:
            snapshot = self._input_snapshot()
        except ValueError:
            self._invalidate()
            raise
        if self._snapshot == snapshot and self._validation_pid == os.getpid():
            return self.validation_report
        self._invalidate()
        started = perf_counter()
        try:
            validated = _validate_cache(self.cache_root, self.manifest_path)
            if snapshot != self._input_snapshot():
                raise ValueError("cache inputs changed during full validation; retry with stable inputs")
        finally:
            self.validation_seconds += perf_counter() - started
        self.meta = validated.meta
        self.records = validated.records
        self.entries = validated.entries
        self.validation_report = validated.report
        self._snapshot = snapshot
        self._validation_pid = os.getpid()
        self.validation_count += 1
        return self.validation_report

    def _invalidate(self) -> None:
        self._snapshot = None
        self._validation_pid = None
        self._arrays.clear()
        self.meta.clear()
        self.records.clear()
        self.entries.clear()
        self.validation_report = {}

    def require_paths(self, cache_root: str | Path, manifest_path: str | Path) -> None:
        if (Path(cache_root).resolve(), Path(manifest_path).resolve()) != (self.cache_root, self.manifest_path):
            raise ValueError("supplied feature store cache/manifest paths do not match")

    def __len__(self) -> int:
        return len(self.entries)

    def utterance_ids(self, *, dataset: str | None = None, split: str | None = None) -> list[str]:
        return [
            utterance_id
            for utterance_id, row in self.records.items()
            if (dataset is None or row["dataset"] == dataset) and (split is None or row["split"] == split)
        ]

    def get(self, utterance_id: str) -> np.ndarray:
        if self._snapshot is None:
            raise ValueError("feature store has no successful validation")
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
    "cleanup_uncommitted_cache_fragments",
    "completed_shards",
    "inspect_cache_resume",
    "load_index",
    "validate_official_cache",
    "validate_cache",
    "validate_success",
    "_atomic_json",
    "_cleanup_partials",
    "_write_index",
]
