"""Privacy-preserving helpers for the IEMOCAP Base downstream notebook.

The notebook only orchestrates these functions.  No helper prints paths,
utterance identifiers, labels for individual samples, or predictions.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from iemocap_downstream.model import BaseModel


CLASS_LABELS = ("ang", "hap", "neu", "sad")
SESSION_IDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.0
    patience: int | None = None
    test_session: int = 5
    validation_session: int | None = None
    encoder_id: str = "emotion2vec_base"


@dataclass(frozen=True)
class PrivatePaths:
    iemocap_root: Path
    work_dir: Path
    checkpoint: Path
    user_dir: Path


@dataclass
class FeatureBundle:
    """Concatenated frame features plus utterance-level metadata."""

    features: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    class_labels: tuple[str, ...] = CLASS_LABELS
    encoder_id: str = "emotion2vec_base"

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def utterance_count(self) -> int:
        return int(len(self.lengths))


class IndexedFeatureDataset(Dataset):
    def __init__(self, bundle: FeatureBundle, indices: Iterable[int]):
        self.bundle = bundle
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        start = int(self.bundle.offsets[index])
        end = start + int(self.bundle.lengths[index])
        return (
            torch.from_numpy(self.bundle.features[start:end].copy()).float(),
            int(self.bundle.labels[index]),
        )

    @staticmethod
    def collate(samples):
        features, labels = zip(*samples)
        lengths = torch.tensor([len(item) for item in features], dtype=torch.long)
        max_length = int(lengths.max())
        batch = features[0].new_zeros((len(features), max_length, features[0].shape[-1]))
        padding_mask = torch.ones((len(features), max_length), dtype=torch.bool)
        for row, feature in enumerate(features):
            batch[row, : len(feature)] = feature
            padding_mask[row, : len(feature)] = False
        return {
            "net_input": {"feats": batch, "padding_mask": padding_mask},
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def private_paths_from_env() -> PrivatePaths:
    names = {
        "iemocap_root": "IEMOCAP_ROOT",
        "work_dir": "IEMOCAP_WORK_DIR",
        "checkpoint": "EMOTION2VEC_CHECKPOINT",
        "user_dir": "EMOTION2VEC_USER_DIR",
    }
    missing = [env_name for env_name in names.values() if not os.environ.get(env_name)]
    if missing:
        raise RuntimeError("Required environment variables are not set: " + ", ".join(missing))
    return PrivatePaths(**{key: Path(os.environ[value]) for key, value in names.items()})


def environment_summary(mode: str, paths: PrivatePaths | None = None) -> dict:
    """Return capability booleans only; never include a private path."""
    result = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "fairseq_available": importlib.util.find_spec("fairseq") is not None,
        "cuda_available": torch.cuda.is_available(),
        "mode": mode,
    }
    if mode == "private":
        if paths is None:
            paths = private_paths_from_env()
        result.update(
            {
                "checkpoint_available": paths.checkpoint.is_file(),
                "iemocap_root_available": paths.iemocap_root.is_dir(),
                "user_dir_available": paths.user_dir.is_dir(),
            }
        )
    return result


def make_demo_bundle(seed: int = 42, input_dim: int = 16, samples_per_class_session: int = 3) -> FeatureBundle:
    """Create deterministic, variable-length synthetic features with five sessions."""
    rng = np.random.default_rng(seed)
    sequences, labels, sessions = [], [], []
    centers = np.eye(len(CLASS_LABELS), input_dim, dtype=np.float32) * 2.5
    for session in SESSION_IDS:
        session_shift = rng.normal(0.0, 0.08, size=input_dim).astype(np.float32)
        for label_id in range(len(CLASS_LABELS)):
            for _ in range(samples_per_class_session):
                length = int(rng.integers(3, 8))
                sequence = centers[label_id] + session_shift + rng.normal(0.0, 0.15, size=(length, input_dim))
                sequences.append(sequence.astype(np.float32))
                labels.append(label_id)
                sessions.append(session)
    lengths = np.asarray([len(item) for item in sequences], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    return FeatureBundle(
        features=np.concatenate(sequences, axis=0),
        lengths=lengths,
        offsets=offsets,
        labels=np.asarray(labels, dtype=np.int64),
        sessions=np.asarray(sessions, dtype=np.int64),
        encoder_id="demo_synthetic_fixed_features",
    )


def load_private_feature_bundle(prefix: str | Path, encoder_id: str = "emotion2vec_base") -> FeatureBundle:
    """Load private artifacts without logging their location or record contents."""
    prefix = Path(prefix)
    features = np.load(str(prefix) + ".npy", allow_pickle=False)
    lengths = np.asarray(
        [int(line) for line in Path(str(prefix) + ".lengths").read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=np.int64,
    )
    raw_rows = [line.split() for line in Path(str(prefix) + ".emo").read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) < 2 for row in raw_rows):
        raise ValueError("The private label file contains a malformed row")
    label_to_id = {label: index for index, label in enumerate(CLASS_LABELS)}
    try:
        labels = np.asarray([label_to_id[row[1]] for row in raw_rows], dtype=np.int64)
        sessions = np.asarray([int(re.search(r"Ses0?([1-5])", row[0]).group(1)) for row in raw_rows], dtype=np.int64)
    except (KeyError, AttributeError) as exc:
        raise ValueError("The private labels must use standard four-class labels and Session 1-5 utterance IDs") from exc
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64) if len(lengths) else np.asarray([], dtype=np.int64)
    return FeatureBundle(features, lengths, offsets, labels, sessions, CLASS_LABELS, encoder_id)


def validate_feature_bundle(bundle: FeatureBundle, expected_input_dim: int | None = None) -> dict:
    """Validate aggregate invariants and return only disclosure-safe totals."""
    errors = []
    if bundle.features.ndim != 2:
        errors.append("features must be a two-dimensional array")
    if not np.issubdtype(bundle.features.dtype, np.number) or not np.isfinite(bundle.features).all():
        errors.append("features must contain finite numeric values")
    if not (len(bundle.lengths) == len(bundle.labels) == len(bundle.sessions)):
        errors.append("length, label, and session row counts must match")
    if len(bundle.lengths) == 0 or np.any(bundle.lengths <= 0):
        errors.append("all utterances must contain at least one frame")
    if int(bundle.lengths.sum()) != len(bundle.features):
        errors.append("the lengths total must equal the number of feature frames")
    if set(np.unique(bundle.sessions).tolist()) != set(SESSION_IDS):
        errors.append("Sessions 1-5 must all be present")
    if np.any(bundle.labels < 0) or np.any(bundle.labels >= len(bundle.class_labels)):
        errors.append("a label is outside the configured class range")
    if expected_input_dim is not None and bundle.features.ndim == 2 and bundle.input_dim != expected_input_dim:
        errors.append("the feature dimension does not match the expected input dimension")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "passed",
        "feature_dim": bundle.input_dim,
        "finite_features": True,
        "total_frames": int(bundle.lengths.sum()),
        "label_rows": int(len(bundle.labels)),
        "sessions_present": list(SESSION_IDS),
    }


def session_split_indices(bundle: FeatureBundle, test_session: int, validation_session: int | None = None) -> dict[str, np.ndarray]:
    if test_session not in SESSION_IDS:
        raise ValueError("test_session must be in 1..5")
    validation_session = validation_session or (test_session % 5 + 1)
    if validation_session not in SESSION_IDS or validation_session == test_session:
        raise ValueError("validation_session must be in 1..5 and differ from test_session")
    split = {
        "train": np.flatnonzero((bundle.sessions != test_session) & (bundle.sessions != validation_session)),
        "validation": np.flatnonzero(bundle.sessions == validation_session),
        "test": np.flatnonzero(bundle.sessions == test_session),
    }
    if any(len(indices) == 0 for indices in split.values()):
        raise ValueError("each split must contain at least one utterance")
    if set(split["test"]) & (set(split["train"]) | set(split["validation"])):
        raise RuntimeError("held-out Session leakage detected")
    return split


def make_session_loaders(bundle: FeatureBundle, config: TrainingConfig) -> tuple[dict[str, DataLoader], int]:
    validation_session = config.validation_session or (config.test_session % 5 + 1)
    indices = session_split_indices(bundle, config.test_session, validation_session)
    loaders = {}
    for name, subset in indices.items():
        dataset = IndexedFeatureDataset(bundle, subset)
        generator = torch.Generator().manual_seed(config.seed)
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            collate_fn=IndexedFeatureDataset.collate,
            generator=generator,
            num_workers=0,
        )
    return loaders, validation_session


def _metrics_from_counts(confusion: np.ndarray) -> dict[str, float]:
    total = int(confusion.sum())
    wa = float(np.trace(confusion) / total) if total else 0.0
    recalls, f1_scores = [], []
    for class_id in range(len(confusion)):
        tp = float(confusion[class_id, class_id])
        actual = float(confusion[class_id, :].sum())
        predicted = float(confusion[:, class_id].sum())
        recalls.append(tp / actual if actual else 0.0)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"wa": wa * 100.0, "ua": float(np.mean(recalls)) * 100.0, "macro_f1": float(np.mean(f1_scores)) * 100.0}


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: str | torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    device = torch.device(device)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_total, sample_total = 0.0, 0
    for batch in loader:
        inputs = batch["net_input"]
        logits = model(inputs["feats"].to(device), inputs["padding_mask"].to(device))
        targets_on_device = batch["labels"].to(device)
        loss_total += float(nn.functional.cross_entropy(logits, targets_on_device, reduction="sum"))
        sample_total += len(targets_on_device)
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = batch["labels"].numpy()
        np.add.at(confusion, (targets, predictions), 1)
    return {"loss": loss_total / sample_total, **_metrics_from_counts(confusion)}


def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    loss_total, sample_total = 0.0, 0
    for batch in loader:
        inputs = batch["net_input"]
        targets = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs["feats"].to(device), inputs["padding_mask"].to(device))
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach()) * len(targets)
        sample_total += len(targets)
    return loss_total / sample_total


def save_classifier_checkpoint(model: BaseModel, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def _safe_torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def reload_and_evaluate(checkpoint_path: str | Path, loader: DataLoader, device: str | torch.device) -> tuple[BaseModel, dict, dict]:
    payload = _safe_torch_load(checkpoint_path)
    metadata = payload["metadata"]
    required = {"encoder_id", "input_dim", "class_labels", "seed", "test_session"}
    if not required.issubset(metadata):
        raise ValueError("checkpoint metadata is incomplete")
    model = BaseModel(
        input_dim=int(metadata["input_dim"]),
        output_dim=len(metadata["class_labels"]),
        hidden_dim=int(metadata.get("hidden_dim", 256)),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    metrics = evaluate_classifier(model, loader, device, len(metadata["class_labels"]))
    return model, metadata, metrics


def _coerce_training_config(config: TrainingConfig | Mapping) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be a TrainingConfig or mapping")
    allowed = set(TrainingConfig.__dataclass_fields__)
    unexpected = set(config) - allowed
    if unexpected:
        raise ValueError("unknown hyperparameters: " + ", ".join(sorted(unexpected)))
    return TrainingConfig(**dict(config))


def _validate_training_config(config: TrainingConfig) -> None:
    if config.epochs < 1 or config.batch_size < 1 or config.hidden_dim < 1:
        raise ValueError("epochs, batch_size, and hidden_dim must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay must be non-negative")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.patience is not None and config.patience < 1:
        raise ValueError("patience must be None or a positive integer")


def _validation_rank(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    """Higher is better: UA, macro F1, then lower validation loss."""
    return (float(metrics["ua"]), float(metrics["macro_f1"]), -float(metrics["loss"]))


def _default_validation_checkpoint(config: TrainingConfig) -> Path:
    payload = json.dumps(asdict(config), ensure_ascii=True, sort_keys=True)
    import hashlib

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return Path("runs") / "iemocap_base_downstream" / f"validation_{digest}.pt"


def run_validation_experiment(
    bundle: FeatureBundle,
    config: TrainingConfig | Mapping,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Train and select a checkpoint using train/validation data only."""
    config = _coerce_training_config(config)
    validate_feature_bundle(bundle)
    _validate_training_config(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    loaders, validation_session = make_session_loaders(bundle, config)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else _default_validation_checkpoint(config)
    model = BaseModel(
        input_dim=bundle.input_dim,
        output_dim=len(bundle.class_labels),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history, best_rank, best_epoch, best_validation, stale_epochs = [], None, 0, None, 0
    metadata = {
        "encoder_id": bundle.encoder_id,
        "input_dim": bundle.input_dim,
        "class_labels": list(bundle.class_labels),
        "seed": config.seed,
        "test_session": config.test_session,
        "validation_session": validation_session,
        "encoder_frozen": True,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "hyperparameters": asdict(config),
        "selection_metric": "validation_ua",
    }
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, loaders["train"], optimizer, criterion, device)
        validation = evaluate_classifier(model, loaders["validation"], device, len(bundle.class_labels))
        row = {"epoch": epoch, "train_loss": train_loss, **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(row)
        rank = _validation_rank(validation)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            best_validation = dict(validation)
            stale_epochs = 0
            checkpoint_metadata = {
                **metadata,
                "best_epoch": best_epoch,
                "selection_metrics": dict(best_validation),
            }
            save_classifier_checkpoint(model, checkpoint_path, checkpoint_metadata)
        else:
            stale_epochs += 1
        if config.patience is not None and stale_epochs >= config.patience:
            break
    optimizer_updated = any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.state_dict().items())
    if not optimizer_updated or not np.isfinite([row["train_loss"] for row in history]).all():
        raise RuntimeError("training did not produce a finite optimizer update")
    return {
        "status": "passed",
        "history": history,
        "validation_metrics": best_validation,
        "best_epoch": best_epoch,
        "model_selection": "best_validation_ua",
        "metadata": {**metadata, "best_epoch": best_epoch, "selection_metrics": dict(best_validation)},
        "hyperparameters": asdict(config),
        "checkpoint_path": str(checkpoint_path),
        "optimizer_updated": optimizer_updated,
    }


def select_best_experiment(experiments: Mapping[str, dict] | Sequence[dict]) -> dict:
    """Select by validation UA, macro F1, loss, then baseline/first order."""
    if isinstance(experiments, Mapping):
        candidates = [(str(name), result) for name, result in experiments.items()]
    else:
        candidates = [(str(result.get("name", index)), result) for index, result in enumerate(experiments)]
    if not candidates:
        raise ValueError("at least one validation experiment is required")
    for name, result in candidates:
        if "validation_metrics" not in result or "checkpoint_path" not in result:
            raise ValueError(f"experiment {name!r} is missing validation results")
    baseline_names = {"base", "baseline", "hp_base"}
    selected_name, selected_result = max(
        candidates,
        key=lambda item: (
            *_validation_rank(item[1]["validation_metrics"]),
            item[0].strip().lower() in baseline_names,
        ),
    )
    return {
        "name": selected_name,
        "result": selected_result,
        "checkpoint_path": selected_result["checkpoint_path"],
        "validation_metrics": dict(selected_result["validation_metrics"]),
        "selection_metrics": dict(selected_result["validation_metrics"]),
        "selection_order": ["validation_ua", "validation_macro_f1", "validation_loss", "baseline_first"],
    }


def evaluate_selected_experiment(bundle: FeatureBundle, selected_experiment: dict) -> dict:
    """Reload the selected checkpoint and evaluate train/validation/test once."""
    result = selected_experiment.get("result", selected_experiment)
    config = _coerce_training_config(result.get("hyperparameters", result["metadata"]["hyperparameters"]))
    device = resolve_device(config.device)
    loaders, _ = make_session_loaders(bundle, config)
    checkpoint_path = selected_experiment.get("checkpoint_path", result["checkpoint_path"])
    payload = _safe_torch_load(checkpoint_path)
    metadata = payload["metadata"]
    model = BaseModel(
        input_dim=int(metadata["input_dim"]),
        output_dim=len(metadata["class_labels"]),
        hidden_dim=int(metadata.get("hidden_dim", 256)),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    split_metrics = {
        split: evaluate_classifier(model, loaders[split], device, len(bundle.class_labels))
        for split in ("train", "validation", "test")
    }
    return {
        "status": "passed",
        "selected_name": selected_experiment.get("name"),
        "split_metrics": split_metrics,
        "metadata": metadata,
        "checkpoint_path": str(checkpoint_path),
    }


def run_one_fold(bundle: FeatureBundle, config: TrainingConfig, checkpoint_path: str | Path) -> dict:
    """Backward-compatible single-fold wrapper around the separated APIs."""
    validation_result = run_validation_experiment(bundle, config, checkpoint_path)
    selected = select_best_experiment({"base": validation_result})
    evaluation = evaluate_selected_experiment(bundle, selected)
    test_metrics = evaluation["split_metrics"]["test"]
    return {
        **validation_result,
        "test_metrics": test_metrics,
        "reload_metrics": dict(test_metrics),
        "split_metrics": evaluation["split_metrics"],
        "metadata": evaluation["metadata"],
    }


def _coerce_fold_experiments(
    config: TrainingConfig | Mapping,
) -> dict[str, TrainingConfig]:
    """Accept one config or a named set of configs without breaking the old API."""
    if isinstance(config, TrainingConfig):
        return {"base": config}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a TrainingConfig, hyperparameter mapping, or named config mapping")
    config_fields = set(TrainingConfig.__dataclass_fields__)
    if set(config).issubset(config_fields):
        return {"base": _coerce_training_config(config)}
    if not config:
        raise ValueError("at least one fold experiment is required")
    experiments = {str(name): _coerce_training_config(value) for name, value in config.items()}
    if len({candidate.seed for candidate in experiments.values()}) != 1:
        raise ValueError("all fold experiments must use the same seed")
    return experiments


def run_five_fold(bundle: FeatureBundle, config: TrainingConfig | Mapping, output_dir: str | Path) -> dict:
    """Run validation-only selection before evaluating each fold's held-out test."""
    output_dir = Path(output_dir)
    experiments = _coerce_fold_experiments(config)
    folds = []
    for session in SESSION_IDS:
        validation_results = {}
        for name, candidate in experiments.items():
            fold_config = TrainingConfig(
                **{**asdict(candidate), "test_session": session, "validation_session": None}
            )
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "experiment"
            checkpoint = output_dir / f"fold_{session}_{safe_name}.pt"
            validation_results[name] = run_validation_experiment(bundle, fold_config, checkpoint)
        selected = select_best_experiment(validation_results)
        evaluation = evaluate_selected_experiment(bundle, selected)
        test_metrics = evaluation["split_metrics"]["test"]
        folds.append(
            {
                "test_session": session,
                "selected_experiment": selected["name"],
                **{name: test_metrics[name] for name in ("wa", "ua", "macro_f1")},
            }
        )
    averages = {name: float(np.mean([fold[name] for fold in folds])) for name in ("wa", "ua", "macro_f1")}
    summary = {"status": "passed", "folds": folds, "average": averages}
    save_json(summary, output_dir / "five_fold_summary.json")
    return summary


def plot_training_history(history: Sequence[dict], output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    for metric, label in (("validation_wa", "WA"), ("validation_ua", "UA"), ("validation_macro_f1", "Macro F1")):
        axes[1].plot(epochs, [row[metric] for row in history], marker="o", label=label)
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Percent", ylim=(0, 100))
    axes[1].legend()
    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_private_manifest_and_labels(paths: PrivatePaths) -> dict:
    """Generate the standard four-class manifest in the user's private environment."""
    import soundfile

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for session in SESSION_IDS:
        evaluation_dir = paths.iemocap_root / f"Session{session}" / "dialog" / "EmoEvaluation"
        for evaluation_file in sorted(evaluation_dir.glob("*.txt")):
            for line in evaluation_file.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split("\t")
                if len(fields) < 3 or not fields[0].startswith("["):
                    continue
                utterance_id, label = fields[1].strip(), fields[2].strip()
                if label not in {"ang", "exc", "hap", "neu", "sad"}:
                    continue
                label = "hap" if label == "exc" else label
                dialog_id = utterance_id.rsplit("_", 1)[0]
                audio = paths.iemocap_root / f"Session{session}" / "sentences" / "wav" / dialog_id / f"{utterance_id}.wav"
                frames = soundfile.info(audio).frames
                rows.append((utterance_id, label, audio.relative_to(paths.iemocap_root), frames))
    if not rows:
        raise RuntimeError("No standard four-class utterances were generated")
    (paths.work_dir / "train.emo").write_text("".join(f"{item[0]}\t{item[1]}\n" for item in rows), encoding="utf-8")
    with (paths.work_dir / "train.tsv").open("w", encoding="utf-8") as handle:
        handle.write(str(paths.iemocap_root) + "\n")
        handle.writelines(f"{item[2]}\t{item[3]}\n" for item in rows)
    return {"status": "passed", "sessions": len(SESSION_IDS), "utterance_count": len(rows)}


def run_private_feature_extraction(
    paths: PrivatePaths,
    project_root: str | Path,
    device: str = "auto",
    layer: int = 0,
) -> dict:
    """Run the extractor while suppressing path-bearing subprocess output."""
    script = Path(project_root) / "iemocap_downstream" / "scripts" / "emotion2vec_speech_features.py"
    command = [
        sys.executable,
        str(script),
        "--data", str(paths.work_dir),
        "--model", str(paths.user_dir),
        "--split", "train",
        "--checkpoint", str(paths.checkpoint),
        "--save-dir", str(paths.work_dir),
        "--layer", str(layer),
        "--device", device,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("Private feature extraction failed; inspect the command locally for details")
    return {"status": "passed", "device": str(resolve_device(device)), "split": "train"}
