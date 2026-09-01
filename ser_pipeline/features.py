"""Restartable extraction of final emotion2vec frame features."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .audio import load_audio_16k_mono, sha256_file
from .cache import (
    CacheIndexEntry,
    _atomic_json,
    _cleanup_partials,
    _write_index,
    completed_shards,
    validate_cache,
    validate_success,
)
from .contracts import (
    CACHE_SCHEMA_VERSION,
    EXTRACTION_CODE_VERSION,
    FEATURE_LAYER,
    normalize_layer,
)
from .manifest import canonical_json, load_manifest, manifest_sha256, validate_manifest_records


DEFAULT_MAX_SHARD_FRAMES = 65536


@dataclass(frozen=True)
class EncoderInfo:
    encoder_name: str
    checkpoint_sha256: str
    feature_dim: int
    feature_layer: str = FEATURE_LAYER


class Emotion2vecEncoder:
    """Lazy fairseq adapter exposing only the supported final representation."""

    def __init__(
        self,
        user_dir: str | Path,
        checkpoint: str | Path,
        *,
        layer: str | int = "final",
        device: str = "auto",
        encoder_name: str = "emotion2vec_base",
        feature_dim: int = 768,
    ):
        normalize_layer(layer)
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        import torch
        import torch.nn.functional as functional
        import fairseq

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        selected = "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"

        @dataclass
        class UserDir:
            user_dir: str

        fairseq.utils.import_user_module(UserDir(str(user_dir)))
        models, _cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        if len(models) != 1:
            raise ValueError("emotion2vec checkpoint must load exactly one model")
        self.model = models[0].eval().to(torch.device(selected))
        self.task = task
        self.device = torch.device(selected)
        self._torch = torch
        self._functional = functional
        self.info = EncoderInfo(
            encoder_name=encoder_name,
            checkpoint_sha256=sha256_file(checkpoint),
            feature_dim=int(feature_dim),
        )

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            source = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(self.device)
            if bool(getattr(self.task.cfg, "normalize", False)):
                source = self._functional.layer_norm(source, source.shape)
            source = source.view(1, -1)
            try:
                result = self.model.extract_features(source, padding_mask=None, remove_extra_tokens=True)
            except TypeError:
                result = self.model.extract_features(source, padding_mask=None)
            if not isinstance(result, Mapping) or "x" not in result:
                raise ValueError("emotion2vec extract_features result must contain 'x'")
            return result["x"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _encoder_info(encoder: Any, expected_dim: int | None) -> EncoderInfo:
    info = getattr(encoder, "info", None)
    if isinstance(info, EncoderInfo):
        if expected_dim is not None and info.feature_dim != expected_dim:
            raise ValueError("encoder feature dimension does not match expected_dim")
        return info
    name = str(getattr(encoder, "encoder_name", "test_or_custom_encoder"))
    checkpoint_hash = str(getattr(encoder, "checkpoint_sha256", "0" * 64))
    dimension = expected_dim if expected_dim is not None else getattr(encoder, "feature_dim", None)
    if dimension is None:
        raise ValueError("expected_dim is required for encoders without EncoderInfo")
    return EncoderInfo(name, checkpoint_hash, int(dimension))


def _extract_array(encoder: Any, waveform: np.ndarray, expected_dim: int) -> np.ndarray:
    output = encoder.extract(waveform) if hasattr(encoder, "extract") else encoder(waveform)
    features = np.asarray(output, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"encoder features must be 2D [frames, dim], got {features.shape}")
    if features.shape[0] <= 0:
        raise ValueError("encoder produced zero frames")
    if features.shape[1] != expected_dim:
        raise ValueError(f"encoder feature dimension must be {expected_dim}, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("encoder features contain non-finite values")
    return features


def _root_for(dataset: str, roots: str | Path | Mapping[str, str | Path]) -> Path:
    if isinstance(roots, Mapping):
        try:
            return Path(roots[dataset])
        except KeyError as exc:
            raise ValueError(f"audio root is missing for dataset: {dataset}") from exc
    return Path(roots)


def _cache_id(meta: Mapping[str, Any]) -> str:
    signature = {key: value for key, value in meta.items() if key not in {"complete", "cache_id"}}
    return hashlib.sha256(canonical_json(signature).encode("utf-8")).hexdigest()[:16]


def _flush_shard(
    split_dir: Path,
    shard_number: int,
    features: list[np.ndarray],
    records: list[Mapping[str, Any]],
    feature_dim: int,
) -> dict[str, Any]:
    if not features or len(features) != len(records):
        raise ValueError("cannot flush an empty or inconsistent shard")
    shard_name = f"shard-{shard_number:05d}.npy"
    index_name = f"shard-{shard_number:05d}.index.jsonl"
    meta_name = f"shard-{shard_number:05d}.meta.json"
    shard_path = split_dir / shard_name
    shard_partial = split_dir / f"{shard_name}.partial"
    array = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    with shard_partial.open("wb") as destination:
        np.save(destination, array, allow_pickle=False)
    shard_partial.replace(shard_path)

    entries: list[CacheIndexEntry] = []
    offset = 0
    for row, current in zip(records, features):
        entries.append(
            CacheIndexEntry(
                dataset=str(row["dataset"]),
                split=str(row["split"]),
                utterance_id=str(row["utterance_id"]),
                shard=shard_name,
                offset=offset,
                num_frames=int(current.shape[0]),
                feature_dim=feature_dim,
                class_index=int(row["class_index"]),
            )
        )
        offset += int(current.shape[0])
    index_path = split_dir / index_name
    _write_index(entries, index_path)
    meta = {
        "shard": shard_name,
        "index": index_name,
        "shard_sha256": sha256_file(shard_path),
        "index_sha256": sha256_file(index_path),
        "frames": int(array.shape[0]),
        "utterances": len(entries),
        "feature_dim": feature_dim,
        "dtype": "float32",
    }
    _atomic_json(meta, split_dir / meta_name)
    return meta


def extract_feature_cache(
    manifest_path: str | Path,
    audio_roots: str | Path | Mapping[str, str | Path],
    cache_root: str | Path,
    encoder: Any,
    *,
    layer: str | int = "final",
    max_shard_frames: int = DEFAULT_MAX_SHARD_FRAMES,
    expected_dim: int | None = None,
) -> dict[str, Any]:
    feature_layer = normalize_layer(layer)
    if max_shard_frames <= 0:
        raise ValueError("max_shard_frames must be positive")
    rows = load_manifest(manifest_path)
    manifest_validation = validate_manifest_records(rows)
    included = [row for row in rows if row["included"]]
    if not included:
        raise ValueError("manifest has no included rows")
    info = _encoder_info(encoder, expected_dim)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    removed_partials = _cleanup_partials(root)
    mapping_versions = sorted({row["mapping_version"] for row in included})
    split_versions = sorted({row["split_version"] for row in included})
    meta: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "encoder_name": info.encoder_name,
        "encoder_checkpoint_sha256": info.checkpoint_sha256,
        "feature_layer": feature_layer,
        "feature_dim": info.feature_dim,
        "dtype": "float32",
        "extraction_code_version": EXTRACTION_CODE_VERSION,
        "git_commit": _git_commit(),
        "manifest_sha256": manifest_sha256(manifest_path),
        "exclusion_contract": manifest_validation["exclusion_contract"],
        "duplicate_audit": manifest_validation["duplicate_audit"],
        "duplicate_exclusion_contract": manifest_validation["duplicate_exclusion_contract"],
        "mapping_versions": mapping_versions,
        "split_versions": split_versions,
        "audio_preprocessing": {
            "target_sample_rate_hz": 16000,
            "channels": "mono_required",
            "resampler": "scipy.signal.resample_poly",
        },
        "shard_policy": {"max_frames_approximately": int(max_shard_frames)},
        "complete": False,
    }
    meta["cache_id"] = _cache_id(meta)
    meta_path = root / "cache_meta.json"
    if not meta_path.exists() and any(root.glob("*/*/shard-*.meta.json")):
        raise ValueError("cannot resume existing shards without cache_meta.json")
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in meta.items():
            if key in {"complete", "git_commit"}:
                continue
            if existing.get(key) != value:
                raise ValueError(f"existing cache metadata mismatch for {key}")
    _atomic_json(meta, meta_path)

    extracted = 0
    skipped = 0
    for dataset, split in sorted({(row["dataset"], row["split"]) for row in included}):
        split_rows = [row for row in included if row["dataset"] == dataset and row["split"] == split]
        split_dir = root / dataset / split
        split_dir.mkdir(parents=True, exist_ok=True)
        success_path = split_dir / "_SUCCESS"
        if success_path.exists():
            validated = validate_success(split_dir)
            observed_ids = [entry.utterance_id for entry in validated["entries"]]
            expected_ids = [row["utterance_id"] for row in split_rows]
            if observed_ids != expected_ids:
                raise ValueError(f"completed cache utterance order mismatch: {dataset}/{split}")
            skipped += len(split_rows)
            continue

        existing_metas, existing_entries = completed_shards(split_dir)
        existing_ids = [entry.utterance_id for entry in existing_entries]
        expected_prefix = [row["utterance_id"] for row in split_rows[: len(existing_ids)]]
        if existing_ids != expected_prefix:
            raise ValueError(f"resume cache utterance prefix mismatch: {dataset}/{split}")
        skipped += len(existing_ids)
        pending = split_rows[len(existing_ids) :]
        shard_number = len(existing_metas)
        shard_features: list[np.ndarray] = []
        shard_rows: list[Mapping[str, Any]] = []
        shard_frames = 0
        new_metas: list[dict[str, Any]] = []
        audio_root = _root_for(dataset, audio_roots)
        for row in pending:
            audio_path = audio_root.joinpath(*Path(row["audio_relpath"]).parts)
            if not audio_path.is_file():
                raise ValueError(f"included audio is missing during extraction: {row['utterance_id']}")
            if sha256_file(audio_path) != row["audio_sha256"]:
                raise ValueError(f"audio hash mismatch during extraction: {row['utterance_id']}")
            waveform = load_audio_16k_mono(audio_path)
            current = _extract_array(encoder, waveform, info.feature_dim)
            if shard_features and shard_frames + current.shape[0] > max_shard_frames:
                new_metas.append(_flush_shard(split_dir, shard_number, shard_features, shard_rows, info.feature_dim))
                shard_number += 1
                shard_features, shard_rows, shard_frames = [], [], 0
            shard_features.append(current)
            shard_rows.append(row)
            shard_frames += int(current.shape[0])
            extracted += 1
        if shard_features:
            new_metas.append(_flush_shard(split_dir, shard_number, shard_features, shard_rows, info.feature_dim))
        all_metas, all_entries = completed_shards(split_dir)
        expected_ids = [row["utterance_id"] for row in split_rows]
        if [entry.utterance_id for entry in all_entries] != expected_ids:
            raise ValueError(f"cache extraction did not cover the split exactly: {dataset}/{split}")
        _atomic_json(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "dataset": dataset,
                "split": split,
                "utterance_count": len(all_entries),
                "shards": all_metas,
            },
            success_path,
        )

    meta["complete"] = True
    _atomic_json(meta, meta_path)
    report = validate_cache(root, manifest_path)
    report.update({"extracted": extracted, "skipped": skipped, "removed_partials": len(removed_partials)})
    return report


extract_features = extract_feature_cache


__all__ = [
    "DEFAULT_MAX_SHARD_FRAMES",
    "Emotion2vecEncoder",
    "EncoderInfo",
    "extract_feature_cache",
    "extract_features",
]
