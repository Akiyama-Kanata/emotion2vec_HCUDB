"""Short real-audio benchmark and formal-run storage/time gates."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio import inspect_audio, load_audio_16k_mono
from .features import Emotion2vecEncoder


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
    free = int(shutil.disk_usage(Path(output_path).resolve()).free)
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


__all__ = [
    "benchmark_audio_extraction",
    "disk_capacity_gate",
    "estimate_full_extraction",
    "save_benchmark",
]
