"""Strict parent/resume checkpoint contracts for the SER decoder."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from .audio import sha256_file
from .contracts import CHECKPOINT_SCHEMA_VERSION, FEATURE_LAYER, LABEL_ORDER
from .model import BaseModel


TRAINING_STAGES = ("msp_train", "hcudb_continue")


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


def _safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
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
    map_location: str | torch.device = "cpu",
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
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    torch.save(payload, partial)
    partial.replace(output)
    return payload


def new_run_id(training_stage: str, seed: int) -> str:
    return f"{training_stage}-seed{seed}-{uuid.uuid4().hex[:12]}"


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
) -> dict[str, Any]:
    payload = load_decoder_checkpoint(
        resume_path,
        expected_signature=expected_signature,
        expected_stage=training_stage,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


__all__ = [
    "TRAINING_STAGES",
    "decoder_signature",
    "load_decoder_checkpoint",
    "new_run_id",
    "restore_parent",
    "restore_resume",
    "save_decoder_checkpoint",
    "validate_signature",
]
