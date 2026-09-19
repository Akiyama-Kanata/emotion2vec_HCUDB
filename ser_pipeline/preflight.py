"""Short real-audio benchmark and formal-run storage/time gates."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .audio import inspect_audio, load_audio_16k_mono
from .audio import sha256_file
from .cache import (
    _atomic_json,
    completed_shards,
    inspect_cache_resume,
)
from .contracts import CACHE_SCHEMA_VERSION, FEATURE_LAYER, label_profile_for_mapping_version
from .features import (
    DEFAULT_MAX_SHARD_FRAMES,
    Emotion2vecEncoder,
    _extract_array,
    _flush_shard,
)
from .manifest import canonical_json, load_manifest, validate_manifest, validate_manifest_records
from .readers import resolved_dataset_root


class FeatureExtractionPreflightError(ValueError):
    """Aggregate every dataset-specific failure found before model construction."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        details = "; ".join(
            f"{dataset}: {item['error']}"
            for dataset, item in report.get("datasets", {}).items()
            if item.get("status") == "error"
        )
        super().__init__(f"feature extraction preflight failed: {details}")


def benchmark_audio_extraction(
    audio_path: str | Path,
    user_dir: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    feature_dim: int = 768,
) -> dict[str, Any]:
    source = Path(audio_path)
    audio = inspect_audio(source, compute_sha256=True)
    load_start = time.perf_counter()
    encoder = Emotion2vecEncoder(
        user_dir,
        checkpoint,
        layer="final",
        device=device,
        feature_dim=feature_dim,
    )
    checkpoint_load_seconds = time.perf_counter() - load_start
    preprocess_start = time.perf_counter()
    waveform = load_audio_16k_mono(source)
    preprocessing_seconds = time.perf_counter() - preprocess_start
    extraction_start = time.perf_counter()
    features = encoder.extract(waveform)
    extraction_seconds = time.perf_counter() - extraction_start
    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] != feature_dim:
        raise ValueError(f"benchmark feature shape is invalid: {features.shape}")
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("benchmark features must be finite float32")
    duration = float(audio["duration_seconds"])
    return {
        "status": "ok",
        "audio_file_name": source.name,
        "audio_sha256": audio["audio_sha256"],
        "source_sample_rate_hz": audio["sample_rate_hz"],
        "source_channels": audio["channels"],
        "source_num_samples": audio["num_samples"],
        "source_duration_seconds": duration,
        "target_sample_rate_hz": 16000,
        "encoder_name": encoder.info.encoder_name,
        "encoder_checkpoint_sha256": encoder.info.checkpoint_sha256,
        "feature_layer": encoder.info.feature_layer,
        "device": str(encoder.device),
        "checkpoint_load_seconds": float(checkpoint_load_seconds),
        "preprocessing_seconds": float(preprocessing_seconds),
        "extraction_seconds": float(extraction_seconds),
        "extraction_realtime_factor": float(extraction_seconds / duration),
        "feature_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_dtype": str(features.dtype),
        "feature_bytes": int(features.nbytes),
        "feature_bytes_per_audio_second": float(features.nbytes / duration),
    }


def estimate_full_extraction(
    total_audio_duration_seconds: float,
    benchmark: dict[str, Any],
    *,
    storage_margin: float = 1.2,
) -> dict[str, Any]:
    if total_audio_duration_seconds <= 0:
        raise ValueError("total_audio_duration_seconds must be positive")
    estimated_seconds = total_audio_duration_seconds * float(benchmark["extraction_realtime_factor"])
    estimated_bytes = total_audio_duration_seconds * float(benchmark["feature_bytes_per_audio_second"])
    return {
        "total_audio_duration_seconds": float(total_audio_duration_seconds),
        "estimated_extraction_seconds": float(estimated_seconds),
        "estimated_feature_bytes": int(round(estimated_bytes)),
        "required_bytes_with_margin": int(round(estimated_bytes * storage_margin)),
        "storage_margin": float(storage_margin),
    }


def disk_capacity_gate(output_path: str | Path, required_bytes_with_margin: int) -> dict[str, Any]:
    target = Path(output_path).resolve()
    while not target.exists() and target.parent != target:
        target = target.parent
    free = int(shutil.disk_usage(target).free)
    required = int(required_bytes_with_margin)
    return {
        "free_bytes": free,
        "required_bytes_with_margin": required,
        "passes": free >= required,
    }


def save_benchmark(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


def _load_parity_report(parity_report: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(parity_report, Mapping):
        payload = dict(parity_report)
    else:
        try:
            payload = json.loads(Path(parity_report).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid parity report: {parity_report}") from exc
    extraction = payload.get("extraction")
    snapshot = payload.get("snapshot")
    if (
        payload.get("passed") is not True
        or payload.get("rtol") != 1e-5
        or payload.get("atol") != 1e-6
        or not isinstance(extraction, dict)
        or not isinstance(snapshot, dict)
        or extraction.get("snapshot") != snapshot
    ):
        raise ValueError("a successful official parity report with matching provenance is required")
    for field in ("extraction_realtime_factor", "feature_bytes_per_audio_second"):
        if not isinstance(payload.get(field), (int, float)) or float(payload[field]) <= 0:
            raise ValueError(f"parity report requires positive {field}")
    return payload


def _expected_official_cache_signature(
    manifest_path: Path,
    rows: list[dict[str, Any]],
    manifest_validation: Mapping[str, Any],
    parity: Mapping[str, Any],
    *,
    expected_dim: int,
    max_shard_frames: int,
) -> dict[str, Any]:
    included = [row for row in rows if bool(row["included"])]
    snapshot = parity["snapshot"]
    extraction = parity["extraction"]
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "encoder_name": "emotion2vec_plus_large",
        "encoder_checkpoint_sha256": snapshot["checkpoint_sha256"],
        "feature_layer": FEATURE_LAYER,
        "feature_dim": int(expected_dim),
        "dtype": "float32",
        "extraction_code_version": extraction["extraction_code_version"],
        "manifest_sha256": manifest_validation["manifest_sha256"],
        "exclusion_contract": manifest_validation["exclusion_contract"],
        "duplicate_audit": manifest_validation["duplicate_audit"],
        "duplicate_exclusion_contract": manifest_validation["duplicate_exclusion_contract"],
        "mapping_versions": sorted({row["mapping_version"] for row in included}),
        "split_versions": sorted({row["split_version"] for row in included}),
        "audio_preprocessing": {
            "target_sample_rate_hz": 16000,
            "channels": "mono_required",
            "resampler": "scipy.signal.resample_poly",
        },
        "shard_policy": {"max_frames_approximately": int(max_shard_frames)},
        "official_provenance": extraction,
    }


def _normalize_preflight_inputs(
    manifest_path: str | Path | Mapping[str, str | Path],
    audio_root: str | Path | Mapping[str, str | Path],
    cache_root: str | Path | Mapping[str, str | Path],
    dataset: str | None,
) -> dict[str, tuple[Path, Path, Path]]:
    values = (manifest_path, audio_root, cache_root)
    if any(isinstance(value, Mapping) for value in values):
        if not all(isinstance(value, Mapping) for value in values):
            raise ValueError("manifest_path, audio_root, and cache_root must all be mappings")
        keys = set(manifest_path)  # type: ignore[arg-type]
        if keys != set(audio_root) or keys != set(cache_root):  # type: ignore[arg-type]
            raise ValueError("preflight dataset mappings must have identical keys")
        return {
            str(name): (
                Path(manifest_path[name]),  # type: ignore[index]
                Path(audio_root[name]),  # type: ignore[index]
                Path(cache_root[name]),  # type: ignore[index]
            )
            for name in sorted(keys)
        }
    manifest = Path(manifest_path)  # type: ignore[arg-type]
    if dataset is None:
        try:
            first = load_manifest(manifest)[0]
            dataset = str(first["dataset"])
        except Exception:
            dataset = manifest.stem
    return {str(dataset): (manifest, Path(audio_root), Path(cache_root))}  # type: ignore[arg-type]


def preflight_feature_extraction(
    manifest_path: str | Path | Mapping[str, str | Path],
    audio_root: str | Path | Mapping[str, str | Path],
    cache_root: str | Path | Mapping[str, str | Path],
    parity_report: str | Path | Mapping[str, Any],
    *,
    dataset: str | None = None,
    expected_dim: int = 1024,
    max_shard_frames: int = DEFAULT_MAX_SHARD_FRAMES,
    storage_margin: float = 1.2,
    capacity_path: str | Path | None = None,
    report_path: str | Path | None = None,
    expected_label_profile: str | None = "official6",
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate all audio and cache resume state before an encoder can be constructed."""
    inputs = _normalize_preflight_inputs(manifest_path, audio_root, cache_root, dataset)
    datasets: dict[str, Any] = {}
    try:
        parity = _load_parity_report(parity_report)
        parity_error = None
    except ValueError as exc:
        parity = None
        parity_error = str(exc)

    required_bytes = 0
    pending_seconds = 0.0
    for name, (manifest, supplied_audio_root, cache) in inputs.items():
        resolved_root: Path | None = None
        try:
            resolved_root = resolved_dataset_root(name, supplied_audio_root).resolve()
            rows = load_manifest(manifest)
            row_datasets = {str(row.get("dataset")) for row in rows}
            if row_datasets != {name}:
                raise ValueError(f"manifest dataset mismatch: expected {name}, got {sorted(row_datasets)}")
            profiles = {label_profile_for_mapping_version(str(row["mapping_version"])) for row in rows}
            if expected_label_profile is not None and profiles != {expected_label_profile}:
                raise ValueError(
                    f"manifest label profile mismatch: expected {expected_label_profile}, got {sorted(profiles)}"
                )
            manifest_report = validate_manifest(
                manifest,
                audio_root=resolved_root,
                audio_root_resolved=True,
            )
            if parity_error is not None:
                raise ValueError(parity_error)
            assert parity is not None
            signature = _expected_official_cache_signature(
                manifest,
                rows,
                manifest_report,
                parity,
                expected_dim=expected_dim,
                max_shard_frames=max_shard_frames,
            )
            resume = inspect_cache_resume(
                cache,
                manifest,
                expected_signature=signature,
                expected_dim=expected_dim,
                max_shard_frames=max_shard_frames,
            )
            duration = 0.0
            included = [row for row in rows if bool(row["included"])]
            by_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in included:
                by_split[(str(row["dataset"]), str(row["split"]))].append(row)
            for split_key, split_rows in by_split.items():
                committed_count = resume["splits"].get(
                    f"{split_key[0]}/{split_key[1]}", {}
                ).get("committed_utterances", 0)
                duration += sum(float(row["duration_seconds"]) for row in split_rows[committed_count:])
            if duration > 0:
                estimate = estimate_full_extraction(duration, parity, storage_margin=storage_margin)
            else:
                estimate = {
                    "total_audio_duration_seconds": 0.0,
                    "estimated_extraction_seconds": 0.0,
                    "estimated_feature_bytes": 0,
                    "required_bytes_with_margin": 0,
                    "storage_margin": float(storage_margin),
                }
            pending_seconds += duration
            required_bytes += int(estimate["required_bytes_with_margin"])
            signature_sha256 = hashlib.sha256(
                canonical_json(signature).encode("utf-8")
            ).hexdigest()
            datasets[name] = {
                "status": "ok",
                "manifest": str(manifest.resolve()),
                "manifest_sha256": manifest_report["manifest_sha256"],
                "label_profile": next(iter(profiles)),
                "resolved_audio_root": str(resolved_root),
                "verified_audio": manifest_report["audio"]["verified_audio"],
                "cache_signature_sha256": signature_sha256,
                "resume": resume,
                "estimate": estimate,
            }
        except Exception as exc:
            datasets[name] = {
                "status": "error",
                "manifest": str(manifest.resolve()),
                "supplied_audio_root": str(supplied_audio_root),
                "resolved_audio_root": str(resolved_root) if resolved_root is not None else None,
                "cache_root": str(cache.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
            }

    gate = disk_capacity_gate(
        capacity_path or next(iter(inputs.values()))[2],
        required_bytes,
    )
    if not gate["passes"]:
        datasets["capacity"] = {
            "status": "error",
            "error": (
                f"insufficient free space: required {required_bytes} bytes, "
                f"available {gate['free_bytes']} bytes"
            ),
        }
    errors = {name: item["error"] for name, item in datasets.items() if item["status"] == "error"}
    report = {
        "status": "error" if errors else "ok",
        "parity_report": str(Path(parity_report).resolve()) if not isinstance(parity_report, Mapping) else "in-memory",
        "datasets": datasets,
        "pending_audio_duration_seconds": pending_seconds,
        "required_bytes_with_margin": required_bytes,
        "capacity": gate,
        "errors": errors,
    }
    if report_path is not None:
        _atomic_json(report, Path(report_path))
    if errors and raise_on_error:
        raise FeatureExtractionPreflightError(report)
    return report


def smoke_test_feature_extraction(
    manifest_path: str | Path,
    audio_root: str | Path,
    cache_root: str | Path,
    encoder: Any,
    *,
    dataset: str | None = None,
    sample_size: int = 10,
    expected_dim: int = 1024,
    max_shard_frames: int = DEFAULT_MAX_SHARD_FRAMES,
) -> dict[str, Any]:
    """Round-trip the first manifest-ordered samples through the real shard writer."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rows = load_manifest(manifest_path)
    validate_manifest_records(rows)
    included = [row for row in rows if bool(row["included"])]
    if dataset is None:
        datasets = {str(row["dataset"]) for row in included}
        if len(datasets) != 1:
            raise ValueError("dataset is required for a multi-dataset manifest")
        dataset = next(iter(datasets))
    sample = [row for row in included if str(row["dataset"]) == dataset][:sample_size]
    if not sample:
        raise ValueError(f"manifest has no included rows for smoke test: {dataset}")
    parent = Path(cache_root).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{Path(cache_root).name}-smoke-", dir=parent))
    try:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sample:
            grouped[str(row["split"])].append(row)
        observed_by_id: dict[str, Any] = {}
        shard_count = 0
        for split, split_rows in grouped.items():
            directory = temporary / str(dataset) / split
            directory.mkdir(parents=True, exist_ok=True)
            shard_number = 0
            feature_rows: list[dict[str, Any]] = []
            feature_arrays: list[np.ndarray] = []
            frame_count = 0
            for row in split_rows:
                path = Path(audio_root).joinpath(*Path(str(row["audio_relpath"])).parts)
                if not path.is_file():
                    raise ValueError(f"smoke audio is missing: {row['utterance_id']}")
                if sha256_file(path) != row["audio_sha256"]:
                    raise ValueError(f"smoke audio hash mismatch: {row['utterance_id']}")
                features = _extract_array(encoder, load_audio_16k_mono(path), expected_dim)
                if feature_arrays and frame_count + int(features.shape[0]) > max_shard_frames:
                    _flush_shard(directory, shard_number, feature_arrays, feature_rows, expected_dim)
                    shard_number += 1
                    feature_rows, feature_arrays, frame_count = [], [], 0
                feature_rows.append(row)
                feature_arrays.append(features)
                frame_count += int(features.shape[0])
            if feature_arrays:
                _flush_shard(directory, shard_number, feature_arrays, feature_rows, expected_dim)
            metas, entries = completed_shards(directory)
            if [entry.utterance_id for entry in entries] != [str(row["utterance_id"]) for row in split_rows]:
                raise ValueError(f"smoke cache utterance order mismatch: {dataset}/{split}")
            for meta in metas:
                array = np.load(directory / meta["shard"], mmap_mode="r", allow_pickle=False)
                if not isinstance(array, np.memmap) or array.ndim != 2 or array.shape[1] != expected_dim:
                    raise ValueError(f"smoke mmap round-trip failed: {meta['shard']}")
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"smoke mmap data contract failed: {meta['shard']}")
            observed_by_id.update({entry.utterance_id: entry for entry in entries})
            shard_count += len(metas)
        sample_ids = [str(row["utterance_id"]) for row in sample]
        if len(observed_by_id) != len(sample_ids) or set(observed_by_id) != set(sample_ids):
            raise ValueError("smoke cache did not cover the manifest sample IDs exactly")
        return {
            "status": "ok",
            "dataset": dataset,
            "sample_count": len(sample),
            "sample_utterance_ids": sample_ids,
            "feature_dim": expected_dim,
            "dtype": "float32",
            "shards": shard_count,
            "temporary_cache_removed": True,
        }
    finally:
        shutil.rmtree(temporary)


__all__ = [
    "benchmark_audio_extraction",
    "FeatureExtractionPreflightError",
    "disk_capacity_gate",
    "estimate_full_extraction",
    "preflight_feature_extraction",
    "save_benchmark",
    "smoke_test_feature_extraction",
]
