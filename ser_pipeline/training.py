"""Dataset-independent decoder training over validated sharded features."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cache import ShardedFeatureStore, _atomic_json
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
from .timing import measure, timed_batches


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
    class_weighting: str = "none"


def training_history_metadata() -> dict[str, Any]:
    """Definitions for recorded epoch metrics, separate from resume loss configuration."""
    return {
        "evaluation": {
            "point": "after_final_optimizer_update_of_epoch",
            "split_order": ["train", "validation"],
            "same_model": True, "mode": "eval", "grad_enabled": False,
        },
        "scores": {
            "uar": "mean recall over the four classes (zero for absent classes)",
            "macro_f1": "mean per-class F1 over the four classes (zero for undefined F1)",
            "accuracy": "fraction of correctly classified utterances; wa is identical",
        },
        "comparison_loss": {
            "fields": ["train.loss", "validation.loss"],
            "class_weighting": "none", "aggregation": "mean_over_all_utterances",
            "formula": "-mean(log(clip(p_true, 1e-12, 1)))", "probability_floor": 1e-12,
        },
        "optimization_loss": {
            "field": "train_loss", "point": "during_optimizer_updates",
            "criterion": "CrossEntropyLoss", "weights": "loss_config.class_weights",
            "batch_reduction": "sum_weighted_nll / sum_observed_label_weights",
            "epoch_aggregation": "unweighted_mean_of_batch_losses_including_final_batch",
        },
        "best_selection": ["validation.uar", "validation.macro_f1", "-validation.loss"],
        "exact_tie": "keep_earlier_epoch",
    }


def training_loss_config(store: ShardedFeatureStore, dataset: str, weighting: str) -> dict[str, Any]:
    """Describe cross entropy using included training utterances only, without reading features."""
    if weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be none or balanced")
    counts = [0] * len(LABEL_ORDER)
    for utterance_id in store.utterance_ids(dataset=dataset, split="train"):
        row = store.records[utterance_id]
        if not row["included"]:
            continue
        counts[int(row["class_index"])] += 1
    total = sum(counts)
    if not total:
        raise ValueError(f"training split is empty: {dataset}")
    if weighting == "balanced" and any(count == 0 for count in counts):
        raise ValueError("balanced class weights require every class in the training split")
    weights = [total / (len(LABEL_ORDER) * count) for count in counts] if weighting == "balanced" else None
    return {
        "name": "cross_entropy",
        "class_weighting": weighting,
        "label_order": list(LABEL_ORDER),
        "train_class_counts": counts,
        "class_weights": weights,
        "reduction": "mean",
    }


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
        features = self.store.get(utterance_id)
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
    shape = (len(features), max_frames, features[0].shape[1])
    # Copy read-only mmap slices directly into the writable batch. Constructing
    # a tensor over a read-only numpy source would expose an unsafe write view.
    numpy_features = isinstance(features[0], np.ndarray)
    batch = torch.zeros(shape, dtype=torch.float32) if numpy_features else features[0].new_zeros(shape)
    batch_array = batch.numpy() if numpy_features else None
    padding_mask = torch.ones((len(features), max_frames), dtype=torch.bool)
    for index, feature in enumerate(features):
        frames = int(feature.shape[0])
        if numpy_features:
            np.copyto(batch_array[index, :frames], feature, casting="no")
        else:
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
        drop_last=False,
        num_workers=0,
    )


def train_one_epoch(model, optimizer, loader, device: torch.device, *, timings=None, class_weights=None) -> float:
    model.train()
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    losses: list[float] = []
    for batch in timed_batches(loader, timings):
        with measure(timings, "compute_seconds", device):
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


def evaluate_loader_metrics(model, loader, device: torch.device, *, timings=None) -> dict[str, Any]:
    """Score a fixed model; restore individual module modes and RNGs even on failure.

    The caller supplies a dedicated unshuffled loader. No optimizer operation or
    gradient clearing occurs here. Loss is the existing unweighted utterance mean.
    """
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_devices = {tensor.device.index for tensor in (*model.parameters(), *model.buffers()) if tensor.is_cuda}
    if torch.device(device).type == "cuda":
        cuda_devices.add(torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device())
    cuda_rng = {index: torch.cuda.get_rng_state(index) for index in cuda_devices}
    generator = getattr(loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    try:
        model.eval()
        truth: list[int] = []
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for batch in timed_batches(loader, timings):
                with measure(timings, "compute_seconds", device):
                    logits = model(
                        batch["net_input"]["feats"].to(device),
                        batch["net_input"]["padding_mask"].to(device),
                    )
                    probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
                    truth.extend(int(value) for value in batch["labels"].tolist())
        if not probabilities:
            raise ValueError("evaluation loader is empty")
        with measure(timings, "metrics_seconds"):
            probs = np.concatenate(probabilities, axis=0)
            return classification_metrics(truth, probs.argmax(axis=1), probs)
    finally:
        # Assign directly: calling train() on parents would overwrite child modes.
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        for index, state in cuda_rng.items():
            torch.cuda.set_rng_state(state, index)
        if generator is not None:
            generator.set_state(loader_rng)


def _prepare_store(cache_root, manifest_path, store, timings):
    previous_validation_seconds = store.validation_seconds if store is not None else 0.0
    with measure(timings, "cache_access_seconds"):
        if store is None:
            store = ShardedFeatureStore(cache_root, manifest_path)
        else:
            store.require_paths(cache_root, manifest_path)
            store.ensure_validated()
    timings["cache_validation_seconds"] = store.validation_seconds - previous_validation_seconds
    return store


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
    store: ShardedFeatureStore | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    timings: dict[str, Any] = {"epochs": []}
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
    if config.class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be none or balanced")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    store = _prepare_store(cache_root, manifest_path, store, timings)
    loss_config = training_loss_config(store, dataset, config.class_weighting)
    history_metadata = training_history_metadata()
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
        resume_payload = restore_resume(
            model, optimizer, resume_checkpoint, signature, training_stage,
            expected_loss_config=loss_config,
        )
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
    # Evaluate the completed epoch's model with a separate loader so scoring
    # cannot advance the shuffled training loader's random generator.
    train_evaluation_loader = make_loader(
        store, dataset, "train", batch_size=config.batch_size, shuffle=False, seed=config.seed
    )
    validation_loader = make_loader(
        store, dataset, "validation", batch_size=config.batch_size, shuffle=False, seed=config.seed
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / f"{training_stage}_seed{config.seed}_best.pt"
    last_path = output / f"{training_stage}_seed{config.seed}_last.pt"
    timing_path = output / f"{training_stage}_seed{config.seed}_timings.json"
    timings["setup_seconds"] = perf_counter() - started - timings["cache_access_seconds"]
    epochs_without_improvement = 0
    print(f"[{dataset} seed={config.seed}] class_weighting={config.class_weighting}", flush=True)
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = perf_counter()
        epoch_timing: dict[str, Any] = {"epoch": epoch, "train": {}, "train_evaluation": {}, "validation": {}}
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, timings=epoch_timing["train"],
            class_weights=loss_config["class_weights"],
        )
        train_evaluation_started = perf_counter()
        train_metrics = evaluate_loader_metrics(
            model, train_evaluation_loader, device, timings=epoch_timing["train_evaluation"]
        )
        epoch_timing["train_evaluation_seconds"] = perf_counter() - train_evaluation_started
        validation_started = perf_counter()
        validation = evaluate_loader_metrics(model, validation_loader, device, timings=epoch_timing["validation"])
        epoch_timing["validation_seconds"] = perf_counter() - validation_started
        history.append({"epoch": epoch, "train_loss": train_loss, "train": train_metrics, "validation": validation})
        save_started = perf_counter()
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
                loss_config=loss_config,
                history_metadata=history_metadata,
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
            loss_config=loss_config,
            history_metadata=history_metadata,
        )
        epoch_timing["save_seconds"] = perf_counter() - save_started
        epoch_timing["total_seconds"] = perf_counter() - epoch_started
        timings["epochs"].append(epoch_timing)
        _atomic_json(timings, timing_path)
        print(
            f"[{dataset} seed={config.seed} epoch={epoch}/{config.epochs} class_weighting={config.class_weighting}]\n"
            f"                 UAR      macro F1\n"
            f"  train          {train_metrics['uar']:.4f}   {train_metrics['macro_f1']:.4f}\n"
            f"  validation     {validation['uar']:.4f}   {validation['macro_f1']:.4f}\n"
            f"  accuracy（参考） train={train_metrics['wa']:.4f}  validation={validation['wa']:.4f}\n"
            f"  best epoch={best_epoch}  選択基準: validation UAR → macro F1 → loss\n"
            f"  time batch={epoch_timing['train']['batch_prepare_seconds']:.2f}s "
            f"train={epoch_timing['train']['compute_seconds']:.2f}s "
            f"train_eval={epoch_timing['train_evaluation_seconds']:.2f}s "
            f"validation={epoch_timing['validation_seconds']:.2f}s "
            f"save={epoch_timing['save_seconds']:.2f}s",
            flush=True,
        )
        if config.patience is not None and epochs_without_improvement >= config.patience:
            break
    if best_metrics is None:
        raise RuntimeError("training did not produce a best validation checkpoint")
    final_save_started = perf_counter()
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
            loss_config=loss_config,
            history_metadata=history_metadata,
        )
    model.load_state_dict(best_state, strict=True)
    timings["finalize_seconds"] = perf_counter() - final_save_started
    timings["total_seconds"] = perf_counter() - started
    _atomic_json(timings, timing_path)
    return {
        "training_stage": training_stage,
        "dataset": dataset,
        "seed": config.seed,
        "device": str(device),
        "best_checkpoint": str(best_path),
        "resume_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "best_training_metrics": next((entry.get("train") for entry in history if entry["epoch"] == best_epoch), None),
        "best_validation_metrics": best_metrics,
        "history": history,
        "parent_checkpoint_id": parent_payload["checkpoint_id"] if parent_payload else None,
        "config": asdict(config),
        "loss_config": loss_config,
        "history_metadata": history_metadata,
        "timings": timings,
        "timings_path": str(timing_path),
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
    store: ShardedFeatureStore | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    timings: dict[str, float] = {}
    selected_device = resolve_device(device)
    store = _prepare_store(cache_root, manifest_path, store, timings)
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
    timings["setup_seconds"] = perf_counter() - started - timings["cache_access_seconds"]
    evaluation_started = perf_counter()
    result = evaluate_model(
        model,
        loader,
        selected_device,
        dataset=dataset,
        split=split,
        set_signature=set_signature,
        timings=timings,
    )
    timings["evaluation_seconds"] = perf_counter() - evaluation_started
    result["checkpoint_id"] = payload["checkpoint_id"]
    result["training_stage"] = payload["training_stage"]
    result["cache_id"] = str(store.meta["cache_id"])
    with measure(timings, "save_seconds"):
        paths = save_evaluation_result(result, output_dir)
    timings["total_seconds"] = perf_counter() - started
    timing_path = Path(output_dir) / "timings.json"
    _atomic_json(timings, timing_path)
    paths["timings"] = str(timing_path)
    print(
        f"[evaluation {dataset}/{split} {payload['training_stage']}] "
        f"uar={result['metrics_4class']['uar']:.4f} "
        f"evaluation={timings['evaluation_seconds']:.2f}s save={timings['save_seconds']:.2f}s "
        f"output={output_dir}", flush=True,
    )
    return {"result": result, "paths": paths, "timings": timings}


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
    "training_loss_config",
    "training_history_metadata",
]
