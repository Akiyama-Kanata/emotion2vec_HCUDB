"""Dataset-independent decoder training over validated sharded features."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cache import ShardedFeatureStore
from .checkpoints import (
    decoder_signature,
    load_decoder_checkpoint,
    new_run_id,
    restore_parent,
    restore_resume,
    save_decoder_checkpoint,
)
from .contracts import LABEL_ORDER
from .evaluation import (
    classification_metrics,
    evaluate_model,
    evaluation_set_signature,
    save_evaluation_result,
)
from .model import BaseModel


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


class CachedFeatureDataset(Dataset):
    def __init__(self, store: ShardedFeatureStore, dataset: str, split: str):
        self.store = store
        self.dataset = dataset
        self.split = split
        self.ids = store.utterance_ids(dataset=dataset, split=split)
        if not self.ids:
            raise ValueError(f"cache dataset split is empty: {dataset}/{split}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        utterance_id = self.ids[index]
        row = self.store.records[utterance_id]
        features = torch.from_numpy(self.store.get(utterance_id).copy()).float()
        return features, int(row["class_index"]), utterance_id


def collate_features(samples):
    if not samples:
        raise ValueError("cannot collate an empty batch")
    features, labels, utterance_ids = zip(*samples)
    if any(feature.ndim != 2 or feature.shape[0] == 0 for feature in features):
        raise ValueError("each cached feature must be non-empty and 2D")
    dimensions = {int(feature.shape[1]) for feature in features}
    if len(dimensions) != 1:
        raise ValueError("batch contains inconsistent feature dimensions")
    max_frames = max(int(feature.shape[0]) for feature in features)
    batch = features[0].new_zeros((len(features), max_frames, features[0].shape[1]))
    padding_mask = torch.ones((len(features), max_frames), dtype=torch.bool)
    for index, feature in enumerate(features):
        frames = int(feature.shape[0])
        batch[index, :frames] = feature
        padding_mask[index, :frames] = False
    return {
        "net_input": {"feats": batch, "padding_mask": padding_mask},
        "labels": torch.tensor(labels, dtype=torch.long),
        "utterance_ids": list(utterance_ids),
    }


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    store: ShardedFeatureStore,
    dataset: str,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CachedFeatureDataset(store, dataset, split),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_features,
    )


def train_one_epoch(model, optimizer, loader, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    for batch in loader:
        features = batch["net_input"]["feats"].to(device)
        mask = batch["net_input"]["padding_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, mask)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("training loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("training loader is empty")
    return float(np.mean(losses))


def evaluate_loader_metrics(model, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    truth: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["net_input"]["feats"].to(device),
                batch["net_input"]["padding_mask"].to(device),
            )
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            truth.extend(int(value) for value in batch["labels"].tolist())
    if not probabilities:
        raise ValueError("evaluation loader is empty")
    probs = np.concatenate(probabilities, axis=0)
    return classification_metrics(truth, probs.argmax(axis=1), probs)


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    loss = float(metrics["loss"])
    return float(metrics["uar"]), float(metrics["macro_f1"]), -loss


def _cpu_state_dict(model: BaseModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_decoder(
    manifest_path: str | Path,
    cache_root: str | Path,
    dataset: str,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    training_stage: str,
    parent_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if parent_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("--parent-checkpoint and --resume-checkpoint are mutually exclusive")
    if training_stage == "msp_train" and dataset != "msp_podcast":
        raise ValueError("msp_train stage requires dataset=msp_podcast")
    if training_stage == "hcudb_continue" and dataset != "hcudb1":
        raise ValueError("hcudb_continue stage requires dataset=hcudb1")
    if training_stage == "hcudb_continue" and parent_checkpoint is None and resume_checkpoint is None:
        raise ValueError("hcudb_continue requires a parent or resume checkpoint")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    store = ShardedFeatureStore(cache_root, manifest_path)
    input_dim = int(store.meta["feature_dim"])
    model = BaseModel(
        input_dim=input_dim,
        output_dim=len(LABEL_ORDER),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    signature = decoder_signature(model, config.seed, store.meta)

    parent_payload = None
    if parent_checkpoint is not None:
        parent_payload = restore_parent(model, parent_checkpoint, signature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    run_id = new_run_id(training_stage, config.seed)
    best_state = _cpu_state_dict(model)
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    preserved_parent_id = None
    preserved_parent_hash = None
    if resume_checkpoint is not None:
        resume_payload = restore_resume(model, optimizer, resume_checkpoint, signature, training_stage)
        start_epoch = int(resume_payload["epoch"]) + 1
        history = list(resume_payload["history"])
        run_id = str(resume_payload["run_id"])
        preserved_parent_id = resume_payload.get("parent_checkpoint_id")
        preserved_parent_hash = resume_payload.get("parent_checkpoint_sha256")
        if resume_payload.get("best_model_state_dict") is not None:
            best_state = copy.deepcopy(resume_payload["best_model_state_dict"])
            best_metrics = dict(resume_payload["best_validation_metrics"])
            best_epoch = int(resume_payload["best_epoch"])
        else:
            best_state = _cpu_state_dict(model)
            best_metrics = dict(resume_payload["validation_metrics"])
            best_epoch = int(resume_payload["epoch"])
    if start_epoch > config.epochs:
        raise ValueError("resume checkpoint epoch is not earlier than configured total epochs")

    train_loader = make_loader(
        store, dataset, "train", batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    validation_loader = make_loader(
        store, dataset, "validation", batch_size=config.batch_size, shuffle=False, seed=config.seed
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / f"{training_stage}_seed{config.seed}_best.pt"
    last_path = output / f"{training_stage}_seed{config.seed}_last.pt"
    epochs_without_improvement = 0
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        validation = evaluate_loader_metrics(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": validation})
        improved = best_metrics is None or selection_key(validation) > selection_key(best_metrics)
        if improved:
            best_metrics = validation
            best_state = _cpu_state_dict(model)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_decoder_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                training_stage=training_stage,
                signature=signature,
                run_id=run_id,
                validation_metrics=validation,
                cache_id=str(store.meta["cache_id"]),
                mapping_versions=list(store.meta["mapping_versions"]),
                split_versions=list(store.meta["split_versions"]),
                parent_checkpoint=parent_checkpoint,
                selection="best_validation",
                best_model_state_dict=best_state,
                best_validation_metrics=best_metrics,
                best_epoch=best_epoch,
                parent_checkpoint_id=preserved_parent_id,
                parent_checkpoint_sha256=preserved_parent_hash,
            )
        else:
            epochs_without_improvement += 1
        save_decoder_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            training_stage=training_stage,
            signature=signature,
            run_id=run_id,
            validation_metrics=validation,
            cache_id=str(store.meta["cache_id"]),
            mapping_versions=list(store.meta["mapping_versions"]),
            split_versions=list(store.meta["split_versions"]),
            parent_checkpoint=parent_checkpoint,
            selection="last",
            best_model_state_dict=best_state,
            best_validation_metrics=best_metrics,
            best_epoch=best_epoch,
            parent_checkpoint_id=preserved_parent_id,
            parent_checkpoint_sha256=preserved_parent_hash,
        )
        if config.patience is not None and epochs_without_improvement >= config.patience:
            break
    if best_metrics is None:
        raise RuntimeError("training did not produce a best validation checkpoint")
    if not best_path.is_file():
        model.load_state_dict(best_state, strict=True)
        save_decoder_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            epoch=best_epoch,
            history=history,
            training_stage=training_stage,
            signature=signature,
            run_id=run_id,
            validation_metrics=best_metrics,
            cache_id=str(store.meta["cache_id"]),
            mapping_versions=list(store.meta["mapping_versions"]),
            split_versions=list(store.meta["split_versions"]),
            parent_checkpoint=parent_checkpoint,
            selection="best_validation",
            best_model_state_dict=best_state,
            best_validation_metrics=best_metrics,
            best_epoch=best_epoch,
            parent_checkpoint_id=preserved_parent_id,
            parent_checkpoint_sha256=preserved_parent_hash,
        )
    model.load_state_dict(best_state, strict=True)
    return {
        "training_stage": training_stage,
        "dataset": dataset,
        "seed": config.seed,
        "device": str(device),
        "best_checkpoint": str(best_path),
        "resume_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "history": history,
        "parent_checkpoint_id": parent_payload["checkpoint_id"] if parent_payload else None,
        "config": asdict(config),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    cache_root: str | Path,
    dataset: str,
    output_dir: str | Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    selected_device = resolve_device(device)
    store = ShardedFeatureStore(cache_root, manifest_path)
    payload = load_decoder_checkpoint(checkpoint_path, map_location=selected_device)
    model_config = dict(payload["signature"]["model_config"])
    model = BaseModel(**model_config).to(selected_device)
    expected = decoder_signature(model, int(payload["signature"]["seed"]), store.meta)
    from .checkpoints import validate_signature

    validate_signature(payload["signature"], expected, context="evaluation")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    loader = make_loader(
        store, dataset, split, batch_size=batch_size, shuffle=False, seed=int(payload["signature"]["seed"])
    )
    set_signature = evaluation_set_signature(manifest_path, dataset, split)
    result = evaluate_model(
        model,
        loader,
        selected_device,
        dataset=dataset,
        split=split,
        set_signature=set_signature,
    )
    result["checkpoint_id"] = payload["checkpoint_id"]
    result["training_stage"] = payload["training_stage"]
    result["cache_id"] = str(store.meta["cache_id"])
    paths = save_evaluation_result(result, output_dir)
    return {"result": result, "paths": paths}


__all__ = [
    "CachedFeatureDataset",
    "TrainingConfig",
    "collate_features",
    "evaluate_checkpoint",
    "evaluate_loader_metrics",
    "make_loader",
    "resolve_device",
    "seed_everything",
    "selection_key",
    "train_decoder",
    "train_one_epoch",
]
