"""Strict audio inspection and 16 kHz mono preprocessing."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 16000


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_audio(path: str | Path, *, compute_sha256: bool = True) -> dict[str, Any]:
    import soundfile as sf

    audio_path = Path(path)
    try:
        info = sf.info(str(audio_path))
    except Exception as exc:
        raise ValueError(f"unreadable audio: {audio_path}") from exc
    if info.frames <= 0:
        raise ValueError(f"audio has zero frames: {audio_path}")
    return {
        "audio_sha256": sha256_file(audio_path) if compute_sha256 else None,
        "audio_size_bytes": int(audio_path.stat().st_size),
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "num_samples": int(info.frames),
        "duration_seconds": float(info.frames / info.samplerate),
    }


def load_audio_16k_mono(path: str | Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio_path = Path(path)
    try:
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"unreadable audio: {audio_path}") from exc
    if waveform.ndim != 1:
        channels = waveform.shape[1] if waveform.ndim == 2 else "unknown"
        raise ValueError(f"audio must be mono, got {channels} channels: {audio_path}")
    if waveform.size == 0:
        raise ValueError(f"audio has zero frames: {audio_path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"audio contains non-finite values: {audio_path}")
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"resampled audio is empty or non-finite: {audio_path}")
    return np.asarray(waveform, dtype=np.float32)
