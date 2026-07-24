"""Reusable building blocks for ``notebooks/audio_to_emotion_vad.ipynb``.

The public functions intentionally do not expose a CLI.  They keep validation,
speaker splitting, feature caching, normalization and folder inference testable
while the notebook remains the only user-facing entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from vad_downstream.data import VADEmotionSpeechDataset
from vad_downstream.model import ParallelEmotionVADClassifier
from vad_downstream.parallel_training import evaluate, train_one_epoch


FEATURE_DIM = 768
VAD_NAMES = ("valence", "arousal", "dominance")
DOMINANCE_WARNING = (
    "Dominance教師がないためDヘッドは未学習です。出力値は研究結果に使用できません。"
)


@dataclass(frozen=True)
class ColumnConfig:
    audio: str = "audio"
    speaker: str = "speaker"
    emotion: str = "emotion"
    valence: str = "valence"
    arousal: str = "arousal"
    dominance: str | None = None


@dataclass
class NotebookConfig:
    audio_dir: str = "data/audio"
    annotation_csv: str = "data/annotations.csv"
    inference_dir: str = "data/inference"
    output_dir: str = "runs/notebooks/audio_to_emotion_vad"
    encoder_checkpoint: str = "artifacts/emotion2vec_base.pt"
    encoder_model_dir: str = "."
    demo_mode: bool = True
    seed: int = 42
    valid_ratio: float = 0.2
    test_ratio: float = 0.2
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    device: str = "cpu"
    columns: ColumnConfig = ColumnConfig()


def validate_wav(path: str | Path) -> dict:
    """Return audio metadata or raise a descriptive validation error."""
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"未対応音声形式です（WAVのみ）: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"音声ファイルがありません: {path}")
    try:
        info = sf.info(path)
    except Exception as exc:
        raise ValueError(f"破損した音声ファイルです: {path}: {exc}") from exc
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError(
            f"16kHzモノラルではありません: {path} "
            f"(sample_rate={info.samplerate}, channels={info.channels})"
        )
    if info.frames <= 0:
        raise ValueError(f"空の音声ファイルです: {path}")
    return {"sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames}


def load_and_validate_annotations(
    csv_path: str | Path,
    audio_dir: str | Path,
    columns: ColumnConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one dataset, infer labels, and validate every referenced WAV."""
    csv_path, audio_dir = Path(csv_path), Path(audio_dir)
    frame = pd.read_csv(csv_path)
    required = [columns.audio, columns.speaker, columns.emotion, columns.valence, columns.arousal]
    if columns.dominance:
        required.append(columns.dominance)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"注釈CSVに必要な列がありません: {missing}")
    if frame.empty:
        raise ValueError("注釈CSVにデータ行がありません")
    for name in (columns.audio, columns.speaker, columns.emotion):
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"列 {name!r} に欠損または空文字があります")
    for name in [columns.valence, columns.arousal] + ([columns.dominance] if columns.dominance else []):
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            bad = frame.loc[values.isna() | ~np.isfinite(values), columns.audio].astype(str).tolist()
            raise ValueError(f"列 {name!r} に不正なVAD値があります: {bad}")
        frame[name] = values.astype(float)
    frame["audio_path"] = frame[columns.audio].map(lambda value: str(audio_dir / str(value)))
    errors = []
    for path in frame["audio_path"]:
        try:
            validate_wav(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("音声検証に失敗しました:\n" + "\n".join(errors))
    labels = sorted(frame[columns.emotion].astype(str).unique().tolist())
    if len(labels) < 2:
        raise ValueError("感情分類には2種類以上のラベルが必要です")
    frame[columns.emotion] = frame[columns.emotion].astype(str)
    return frame, labels


def speaker_split(
    frame: pd.DataFrame,
    speaker_column: str,
    emotion_column: str,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic, speaker-disjoint train/valid/test assignments."""
    if valid_ratio < 0 or test_ratio < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio/test_ratio は0以上で、合計を1未満にしてください")
    speakers = sorted(frame[speaker_column].astype(str).unique())
    if len(speakers) < 3:
        raise ValueError("話者単位の3分割には3話者以上が必要です")
    labels = set(frame[emotion_column].astype(str))
    rng = random.Random(seed)
    assignment = None
    for _ in range(2000):
        shuffled = speakers.copy()
        rng.shuffle(shuffled)
        n_valid = max(1, round(len(shuffled) * valid_ratio))
        n_test = max(1, round(len(shuffled) * test_ratio))
        if n_valid + n_test >= len(shuffled):
            n_valid = n_test = 1
        split_for = {speaker: "train" for speaker in shuffled}
        for speaker in shuffled[:n_valid]:
            split_for[speaker] = "valid"
        for speaker in shuffled[n_valid : n_valid + n_test]:
            split_for[speaker] = "test"
        train_labels = set(
            frame.loc[frame[speaker_column].astype(str).map(split_for) == "train", emotion_column].astype(str)
        )
        if train_labels == labels:
            assignment = split_for
            break
    if assignment is None:
        raise ValueError("学習splitに全感情ラベルを含む話者分割を作成できません")
    result = frame.copy()
    result["split"] = result[speaker_column].astype(str).map(assignment)
    assert_no_speaker_leakage(result, speaker_column)
    missing = labels - set(result.loc[result["split"] == "train", emotion_column])
    if missing:
        raise ValueError(f"学習splitに感情ラベルがありません: {sorted(missing)}")
    return result


def assert_no_speaker_leakage(frame: pd.DataFrame, speaker_column: str) -> None:
    counts = frame.groupby(speaker_column)["split"].nunique()
    leaking = counts[counts > 1].index.astype(str).tolist()
    if leaking:
        raise ValueError(f"複数splitに含まれる話者があります: {leaking}")


@dataclass
class TrainMinMaxNormalizer:
    minimum: dict[str, float]
    maximum: dict[str, float]
    trained: dict[str, bool]

    @classmethod
    def fit(cls, train: pd.DataFrame, column_map: Mapping[str, str | None]):
        minimum, maximum, trained = {}, {}, {}
        for name in VAD_NAMES:
            column = column_map.get(name)
            trained[name] = bool(column and column in train.columns)
            if trained[name]:
                values = train[column].astype(float).to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"{name} に有限でない学習値があります")
                minimum[name], maximum[name] = float(values.min()), float(values.max())
            else:
                minimum[name] = maximum[name] = 0.0
        return cls(minimum, maximum, trained)

    def transform_value(self, name: str, value: float) -> float:
        if not self.trained[name]:
            return 0.0
        low, high = self.minimum[name], self.maximum[name]
        return 0.0 if high == low else 2.0 * (float(value) - low) / (high - low) - 1.0

    def inverse_value(self, name: str, value: float) -> float | None:
        if not self.trained[name]:
            return None
        low, high = self.minimum[name], self.maximum[name]
        return low if high == low else (float(value) + 1.0) * (high - low) / 2.0 + low

    def transform_frame(self, frame: pd.DataFrame, column_map: Mapping[str, str | None]):
        result = frame.copy()
        for name in VAD_NAMES:
            column = column_map.get(name)
            result[f"{name}_normalized"] = (
                result[column].map(lambda value: self.transform_value(name, value))
                if self.trained[name]
                else 0.0
            )
        return result

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping):
        return cls(
            {key: float(value) for key, value in payload["minimum"].items()},
            {key: float(value) for key, value in payload["maximum"].items()},
            {key: bool(value) for key, value in payload["trained"].items()},
        )


class FeatureCache:
    """Cache frame features by WAV identity and encoder checkpoint identity."""

    def __init__(self, cache_dir: str | Path, encoder_id: str, extractor: Callable[[Path], np.ndarray]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_id = str(encoder_id)
        self.extractor = extractor

    def key(self, wav_path: str | Path) -> str:
        path = Path(wav_path).resolve()
        stat = path.stat()
        identity = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.encoder_id}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def get(self, wav_path: str | Path) -> np.ndarray:
        validate_wav(wav_path)
        target = self.cache_dir / f"{self.key(wav_path)}.npy"
        if target.exists():
            try:
                features = np.load(target, allow_pickle=False)
                _validate_features(features, target)
                return features.astype(np.float32, copy=False)
            except (OSError, ValueError):
                target.unlink(missing_ok=True)
        features = np.asarray(self.extractor(Path(wav_path)), dtype=np.float32)
        _validate_features(features, wav_path)
        np.save(target, features, allow_pickle=False)
        return features


def demo_feature_extractor(wav_path: Path) -> np.ndarray:
    """Deterministic 768-D fake extractor used only for end-to-end demos."""
    wav, _ = sf.read(wav_path, dtype="float32")
    frame_size = 400
    count = max(1, math.ceil(len(wav) / frame_size))
    padded = np.pad(wav, (0, count * frame_size - len(wav))).reshape(count, frame_size)
    spectrum = np.abs(np.fft.rfft(padded, n=1534, axis=1))[:, :FEATURE_DIM]
    scale = np.maximum(spectrum.max(axis=1, keepdims=True), 1e-8)
    return (spectrum / scale).astype(np.float32)


def make_emotion2vec_extractor(encoder, device="cpu", normalize_audio=False):
    """Wrap a loaded/frozen emotion2vec encoder as a cache extractor."""
    device = torch.device(device)
    encoder.to(device).eval()

    def extract(path: Path) -> np.ndarray:
        wav, _ = sf.read(path, dtype="float32")
        source = torch.from_numpy(wav).to(device)
        if normalize_audio:
            source = torch.nn.functional.layer_norm(source, source.shape)
        with torch.no_grad():
            try:
                output = encoder.extract_features(source.unsqueeze(0), padding_mask=None, mask=False, remove_extra_tokens=True)
            except TypeError:
                output = encoder.extract_features(source.unsqueeze(0), padding_mask=None)
        features = output["x"] if isinstance(output, dict) else output[0] if isinstance(output, tuple) else output
        return features.squeeze(0).detach().cpu().numpy()

    return extract


def make_dataset(frame: pd.DataFrame, labels: Sequence[str], cache: FeatureCache, columns: ColumnConfig):
    """Build the existing collatable dataset from cached variable-length features."""
    feature_rows = [cache.get(path) for path in frame["audio_path"]]
    lengths = np.asarray([len(item) for item in feature_rows], dtype=np.int64)
    features = np.concatenate(feature_rows, axis=0)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    label_to_id = {label: index for index, label in enumerate(labels)}
    vad = frame[["valence_normalized", "arousal_normalized", "dominance_normalized"]].to_numpy(np.float32)
    masks = np.ones_like(vad, dtype=np.bool_)
    if columns.dominance is None:
        masks[:, 2] = False
    return VADEmotionSpeechDataset(
        features, lengths, offsets, vad,
        frame[columns.emotion].map(label_to_id).to_numpy(np.int64),
        utt_ids=frame[columns.audio].astype(str).tolist(),
        emotion_labels=frame[columns.emotion].astype(str).tolist(),
        class_labels=list(labels), vad_target_masks=masks,
    )


def make_loader(dataset, batch_size=8, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=dataset.collator)


def train_epochs(model, train_loader, valid_loader, epochs, learning_rate, device, labels):
    """Train heads and return JSON-serializable per-epoch history."""
    optimizer = torch.optim.AdamW(model.task_parameters(include_dominance=any(
        batch["vad_target_mask"][:, 2].any().item() for batch in train_loader
    )), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, optimizer, train_loader, device)
        valid_metrics = evaluate(model, valid_loader, device, class_labels=list(labels))
        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
    return history


def predict_wav_folder(
    folder: str | Path,
    model: ParallelEmotionVADClassifier,
    labels: Sequence[str],
    normalizer: TrainMinMaxNormalizer,
    cache: FeatureCache,
    output_csv: str | Path,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Validate and infer every WAV in a folder, then persist a flat CSV."""
    paths = sorted(Path(folder).glob("*.wav"))
    if not paths:
        raise ValueError(f"推論フォルダにWAVがありません: {folder}")
    errors = []
    for path in paths:
        try:
            validate_wav(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("推論音声の検証に失敗しました:\n" + "\n".join(errors))
    model.eval()
    rows = []
    dominance_status = "trained" if normalizer.trained["dominance"] else "untrained"
    with torch.no_grad():
        for path in paths:
            feature = torch.from_numpy(cache.get(path)).unsqueeze(0).to(device)
            output = model(feature)
            probabilities = torch.softmax(output["logits"][0], dim=0).cpu().numpy()
            vad = output["vad"][0].cpu().numpy()
            row = {"audio_file": path.name, "predicted_emotion": labels[int(probabilities.argmax())]}
            row.update({f"probability_{label}": float(probabilities[i]) for i, label in enumerate(labels)})
            for index, name in enumerate(VAD_NAMES):
                row[f"{name}_normalized"] = float(vad[index])
                restored = normalizer.inverse_value(name, float(vad[index]))
                # Without D labels there is no defensible original-scale inverse.
                # Keep the requested numeric schema by repeating the raw head value
                # and make that limitation machine-readable as well as warned.
                row[name] = float(vad[index]) if restored is None else restored
                row[f"{name}_status"] = "trained" if normalizer.trained[name] else "untrained"
                row[f"{name}_scale_status"] = (
                    "original_scale" if restored is not None else "unavailable_normalized_value_repeated"
                )
            row["dominance_status"] = dominance_status
            row["warning"] = DOMINANCE_WARNING if dominance_status == "untrained" else ""
            rows.append(row)
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if dominance_status == "untrained":
        warnings.warn(DOMINANCE_WARNING, RuntimeWarning, stacklevel=2)
    return result


def save_json(payload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_features(features: np.ndarray, source) -> None:
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"特徴量は [T, {FEATURE_DIM}] である必要があります: {source}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"特徴量に有限でない値があります: {source}")
