"""Strict parent/resume checkpoint contracts for the SER decoder."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from .audio import sha256_file
from .contracts import CHECKPOINT_SCHEMA_VERSION, FEATURE_LAYER, LABEL_ORDER, OFFICIAL_TARGET_ORDER
from .contracts import OFFICIAL_CHECKPOINT_SCHEMA_VERSION
from .model import BaseModel


TRAINING_STAGES = ("msp_train", "hcudb_continue")
_UNSET = object()


def decoder_signature(model: BaseModel, seed: int, cache_meta: Mapping[str, Any]) -> dict[str, Any]:
    if cache_meta.get("feature_layer") != FEATURE_LAYER:
        raise ValueError("decoder requires final_after_encoder_norm cache features")
    if int(cache_meta.get("feature_dim", -1)) != model.input_dim:
        raise ValueError("decoder input_dim does not match cache feature_dim")
    return {
        "label_order": list(LABEL_ORDER),
        "model_type": "BaseModel",
        "model_config": {
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout_probability,
        },
        "input_dim": model.input_dim,
        "seed": int(seed),
        "encoder_signature": {
            "encoder_name": cache_meta.get("encoder_name"),
            "encoder_checkpoint_sha256": cache_meta.get("encoder_checkpoint_sha256"),
            "feature_layer": cache_meta.get("feature_layer"),
        },
    }


def validate_signature(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    keys = ("label_order", "model_type", "model_config", "input_dim", "seed", "encoder_signature")
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"{context} checkpoint signature mismatch for {key}")


def _safe_torch_load(path: Path, map_location: str | torch.device | None = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch 1.x compatibility
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("decoder checkpoint must contain a dictionary")
    return payload


def load_decoder_checkpoint(
    path: str | Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
    expected_stage: str | None = None,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    payload = _safe_torch_load(checkpoint_path, map_location=map_location)
    required = {
        "checkpoint_schema_version",
        "checkpoint_id",
        "training_stage",
        "signature",
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "history",
        "run_id",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"decoder checkpoint is missing fields: {missing}")
    if payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("decoder checkpoint schema mismatch")
    if payload["training_stage"] not in TRAINING_STAGES:
        raise ValueError("unknown decoder training_stage")
    if expected_stage is not None and payload["training_stage"] != expected_stage:
        raise ValueError("resume checkpoint training_stage mismatch")
    if expected_signature is not None:
        validate_signature(payload["signature"], expected_signature, context="decoder")
    return payload


def save_decoder_checkpoint(
    path: str | Path,
    *,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    training_stage: str,
    signature: Mapping[str, Any],
    run_id: str,
    validation_metrics: Mapping[str, Any],
    cache_id: str,
    mapping_versions: list[str],
    split_versions: list[str],
    parent_checkpoint: str | Path | None = None,
    selection: str = "last",
    best_model_state_dict: Mapping[str, Any] | None = None,
    best_validation_metrics: Mapping[str, Any] | None = None,
    best_epoch: int | None = None,
    parent_checkpoint_id: str | None = None,
    parent_checkpoint_sha256: str | None = None,
    loss_config: Mapping[str, Any] | None = None,
    history_metadata: Mapping[str, Any] | None = None,
    monitoring_config: Mapping[str, Any] | None | object = _UNSET,
    train_monitoring: Mapping[str, Any] | object = _UNSET,
) -> dict[str, Any]:
    if training_stage not in TRAINING_STAGES:
        raise ValueError(f"invalid training_stage: {training_stage}")
    parent_id = None
    parent_hash = None
    if parent_checkpoint is not None:
        parent_path = Path(parent_checkpoint)
        parent_payload = load_decoder_checkpoint(parent_path)
        parent_id = parent_payload["checkpoint_id"]
        parent_hash = sha256_file(parent_path)
    elif parent_checkpoint_id is not None or parent_checkpoint_sha256 is not None:
        if not parent_checkpoint_id or not parent_checkpoint_sha256:
            raise ValueError("preserved parent checkpoint ID and SHA-256 must be provided together")
        parent_id = str(parent_checkpoint_id)
        parent_hash = str(parent_checkpoint_sha256)
    identity = {
        "run_id": run_id,
        "epoch": int(epoch),
        "training_stage": training_stage,
        "selection": selection,
        "parent_checkpoint_id": parent_id,
    }
    checkpoint_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "training_stage": training_stage,
        "signature": dict(signature),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "history": list(history),
        "run_id": str(run_id),
        "validation_metrics": dict(validation_metrics),
        "selection": selection,
        "cache_id": cache_id,
        "mapping_versions": list(mapping_versions),
        "split_versions": list(split_versions),
        "parent_checkpoint_id": parent_id,
        "parent_checkpoint_sha256": parent_hash,
        "best_model_state_dict": dict(best_model_state_dict) if best_model_state_dict is not None else None,
        "best_validation_metrics": dict(best_validation_metrics) if best_validation_metrics is not None else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_training_metrics": next((entry.get("train") for entry in history if entry.get("epoch") == best_epoch), None),
        "loss_config": dict(loss_config) if loss_config is not None else None,
    }
    if history_metadata is not None:
        payload["history_metadata"] = dict(history_metadata)
    if monitoring_config is not _UNSET:
        payload["monitoring_config"] = dict(monitoring_config) if monitoring_config is not None else None
    if train_monitoring is not _UNSET:
        payload["train_monitoring"] = dict(train_monitoring)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    torch.save(payload, partial)
    partial.replace(output)
    return payload


def update_decoder_checkpoint_results(
    path: str | Path,
    *,
    best_training_metrics: Mapping[str, Any],
    best_train_evaluation_seconds: float,
    best_train_evaluation_reused_from_monitor: bool,
) -> dict[str, Any]:
    """Atomically add final train results without rebuilding checkpoint state."""
    output = Path(path)
    payload = load_decoder_checkpoint(output, map_location=None)
    payload["best_training_metrics"] = dict(best_training_metrics)
    payload["best_train_evaluation_seconds"] = float(best_train_evaluation_seconds)
    payload["best_train_evaluation_reused_from_monitor"] = bool(
        best_train_evaluation_reused_from_monitor
    )
    partial = output.with_name(output.name + ".partial")
    torch.save(payload, partial)
    partial.replace(output)
    return payload


def new_run_id(training_stage: str, seed: int) -> str:
    return f"{training_stage}-seed{seed}-{uuid.uuid4().hex[:12]}"


def capture_rng_state(generator):
    """Serialize Python, NumPy, Torch and loader RNGs without arbitrary globals."""
    import random
    import numpy as np
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng_state(state, generator):
    import random
    import numpy as np
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("D resume CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def _validate_official_payload(payload, official_head):
    from .model import OfficialHeadModel
    from .training import OfficialRowGuard, selection_key

    required = {"checkpoint_id", "signature", "model_state_dict", "optimizer_state_dict", "rng_state", "config", "seed", "epoch",
                "history", "run_id", "validation_metrics", "selection", "best_checkpoint", "best_epoch",
                "best_model_state_dict", "best_validation_metrics", "loss_config", "train_monitoring", "monitoring_config"}
    if not required <= payload.keys() or payload.get("checkpoint_schema_version") != OFFICIAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("D checkpoint schema mismatch or missing fields")
    if payload.get("training_stage") != "hcudb_official_continue":
        raise ValueError("D checkpoint training stage mismatch")
    if payload.get("selection") not in {"last", "best_validation"}:
        raise ValueError("D checkpoint selection mismatch")
    identity = f"{payload['run_id']}:{payload['epoch']}:{payload['selection']}"
    if payload["checkpoint_id"] != hashlib.sha256(identity.encode()).hexdigest()[:20]:
        raise ValueError("D checkpoint identity mismatch")
    if payload.get("official_snapshot") != official_head.provenance or payload.get("label_spec") != official_head.label_spec.as_dict():
        raise ValueError("D checkpoint official snapshot/labels mismatch")
    if payload.get("protection") != OfficialRowGuard.method:
        raise ValueError("D row protection metadata missing or invalid")
    signature = payload["signature"]
    if (signature.get("model_type") != "OfficialHeadModel" or signature.get("condition") != "D"
            or signature.get("label_order") != list(OFFICIAL_TARGET_ORDER)
            or signature.get("official_snapshot") != official_head.provenance
            or signature.get("label_spec") != official_head.label_spec.as_dict()
            or signature.get("protection") != OfficialRowGuard.method):
        raise ValueError("D checkpoint signature mismatch")
    config = payload["config"]
    if (config.get("weight_decay") != 0 or config.get("class_weighting") != "none"
            or config.get("patience") is not None or config.get("dropout") != 0 or config.get("seed") != payload["seed"]):
        raise ValueError("D checkpoint training configuration mismatch")
    loss_config = payload.get("loss_config")
    if (not isinstance(loss_config, dict)
            or loss_config.get("name") != "cross_entropy"
            or loss_config.get("logit_space") != "official9"
            or loss_config.get("class_weighting") != "none"
            or loss_config.get("label_smoothing") != 0.0
            or loss_config.get("target_names") != list(OFFICIAL_TARGET_ORDER)
            or loss_config.get("target_indices") != list(official_head.label_spec.target_indices)
            or loss_config.get("class_weights") is not None):
        raise ValueError("D checkpoint must use unweighted official9 cross entropy")
    model = OfficialHeadModel(official_head, condition="D").cpu()
    guard = OfficialRowGuard(model, official_head)

    def validate_epoch(record):
        guard.validate_state(record["model_state_dict"])
        model.load_state_dict(record["model_state_dict"], strict=True)
        optimizer = torch.optim.AdamW([model.proj.weight, model.proj.bias], lr=config["learning_rate"], weight_decay=0)
        expected_groups = optimizer.state_dict()["param_groups"]
        if record["optimizer_state_dict"]["param_groups"] != expected_groups:
            raise ValueError("D checkpoint optimizer configuration mismatch")
        optimizer.load_state_dict(record["optimizer_state_dict"])
        if len(optimizer.state) != 2:
            raise ValueError("D checkpoint must contain both Adam parameter states")
        guard.validate(optimizer)
        if not {"python", "numpy", "torch", "cuda", "loader"} <= record["rng_state"].keys():
            raise ValueError("D checkpoint RNG state incomplete")

    validate_epoch(payload)
    best = payload["best_checkpoint"]
    validate_epoch(best)
    guard.validate_state(payload["best_model_state_dict"])
    if (best["epoch"] != payload["best_epoch"] or best["validation_metrics"] != payload["best_validation_metrics"]
            or best["signature"] != signature or best["run_id"] != payload["run_id"]):
        raise ValueError("D checkpoint best history mismatch")
    if best.get("selection") != "best_validation" or best.get("config") != config:
        raise ValueError("D checkpoint best selection/configuration mismatch")
    if any(not torch.equal(value.cpu(), best["model_state_dict"][key].cpu()) for key, value in payload["best_model_state_dict"].items()):
        raise ValueError("D checkpoint best state mismatch")
    history_epochs = [row["epoch"] for row in payload["history"]]
    if not history_epochs or history_epochs != list(range(1, history_epochs[-1] + 1)):
        raise ValueError("D checkpoint epoch history is not contiguous")
    if payload["epoch"] not in history_epochs:
        raise ValueError("D checkpoint epoch is missing from history")
    current = payload["history"][payload["epoch"] - 1]
    if current["validation"] != payload["validation_metrics"]:
        raise ValueError("D checkpoint current validation metrics mismatch")
    selected = max(payload["history"], key=lambda row: selection_key(row["validation"]))
    if selected["epoch"] != best["epoch"] or selected["validation"] != best["validation_metrics"]:
        raise ValueError("D checkpoint best selection mismatch")


def save_official_checkpoint(path, payload, official_head):
    """Atomically persist a validated D artifact; C never uses this schema."""
    payload = dict(payload, checkpoint_schema_version=OFFICIAL_CHECKPOINT_SCHEMA_VERSION)
    identity = f"{payload['run_id']}:{payload['epoch']}:{payload['selection']}"
    payload["checkpoint_id"] = hashlib.sha256(identity.encode()).hexdigest()[:20]
    _validate_official_payload(payload, official_head)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    torch.save(payload, partial)
    partial.replace(output)


def load_official_checkpoint(path, official_head):
    payload = _safe_torch_load(Path(path))
    _validate_official_payload(payload, official_head)
    return payload


def restore_parent(
    model: BaseModel,
    parent_path: str | Path,
    expected_signature: Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_decoder_checkpoint(parent_path, expected_signature=expected_signature)
    if payload["training_stage"] != "msp_train":
        raise ValueError("HCUDB parent checkpoint must have training_stage=msp_train")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def restore_resume(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    resume_path: str | Path,
    expected_signature: Mapping[str, Any],
    training_stage: str,
    *,
    expected_loss_config: Mapping[str, Any] | None = None,
    expected_monitoring_config: Mapping[str, Any] | None | object = _UNSET,
) -> dict[str, Any]:
    payload = load_decoder_checkpoint(
        resume_path,
        expected_signature=expected_signature,
        expected_stage=training_stage,
    )
    if expected_loss_config is not None:
        saved_loss = payload.get("loss_config")
        if saved_loss is None:
            # Older checkpoints used unweighted cross entropy exclusively.
            if expected_loss_config["class_weighting"] != "none":
                raise ValueError("resume checkpoint loss configuration mismatch: legacy unweighted loss")
        elif saved_loss != dict(expected_loss_config):
            raise ValueError("resume checkpoint loss configuration mismatch")
    if expected_monitoring_config is not _UNSET:
        if "monitoring_config" not in payload:
            if expected_monitoring_config is not None:
                raise ValueError("resume checkpoint monitoring configuration mismatch: legacy full-train monitoring")
        else:
            expected = dict(expected_monitoring_config) if expected_monitoring_config is not None else None
            if payload["monitoring_config"] != expected:
                raise ValueError("resume checkpoint monitoring configuration mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


__all__ = [
    "TRAINING_STAGES",
    "capture_rng_state",
    "decoder_signature",
    "load_decoder_checkpoint",
    "load_official_checkpoint",
    "new_run_id",
    "restore_parent",
    "restore_resume",
    "restore_rng_state",
    "save_decoder_checkpoint",
    "save_official_checkpoint",
    "update_decoder_checkpoint_results",
    "validate_signature",
]
