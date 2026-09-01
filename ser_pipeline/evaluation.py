"""Four-class metrics, IEMOCAP three-class summary, and result persistence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .contracts import LABEL_ORDER, RESULT_LIMITATIONS, RESULT_SCHEMA_VERSION
from .manifest import load_manifest, manifest_sha256, validate_manifest_records


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 4) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        if not 0 <= int(truth) < num_classes or not 0 <= int(prediction) < num_classes:
            raise ValueError("class index is outside the decoder label order")
        matrix[int(truth), int(prediction)] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray | None = None,
    *,
    reported_classes: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("y_true and y_pred must be non-empty equal-length vectors")
    matrix = confusion_matrix(truth, prediction, len(LABEL_ORDER))
    class_rows = []
    recalls = []
    f1_values = []
    for index in reported_classes:
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append(
            {
                "class_index": int(index),
                "class_label": LABEL_ORDER[index],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
        recalls.append(recall)
        f1_values.append(f1)
    accuracy = float(np.mean(truth == prediction))
    loss = None
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (len(truth), len(LABEL_ORDER)):
            raise ValueError("probabilities must have shape [samples, 4]")
        if not np.isfinite(probs).all() or np.any(probs < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each probability row must sum to one")
        loss = float(-np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)).mean())
    return {
        "accuracy": accuracy,
        "wa": accuracy,
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "loss": loss,
        "reported_class_indices": [int(value) for value in reported_classes],
        "class_metrics": class_rows,
        "confusion_matrix": matrix.tolist(),
    }


def build_evaluation_result(
    utterance_ids: Sequence[str],
    y_true: Sequence[int],
    probabilities: np.ndarray,
    *,
    dataset: str,
    split: str,
    set_signature: Mapping[str, Any],
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    predictions = probs.argmax(axis=1).astype(np.int64)
    if len(utterance_ids) != len(y_true) or len(y_true) != len(probs):
        raise ValueError("evaluation ids, labels, and probabilities must align")
    metrics_4class = classification_metrics(y_true, predictions, probs)
    primary_3class = None
    if dataset == "iemocap":
        truth = np.asarray(y_true, dtype=np.int64)
        mask = truth != 3
        primary_3class = classification_metrics(
            truth[mask],
            predictions[mask],
            probs[mask],
            reported_classes=(0, 1, 2),
        )
    prediction_rows = []
    for utterance_id, truth, prediction, probability in zip(utterance_ids, y_true, predictions, probs):
        prediction_rows.append(
            {
                "utterance_id": str(utterance_id),
                "true_class_index": int(truth),
                "true_label": LABEL_ORDER[int(truth)],
                "predicted_class_index": int(prediction),
                "predicted_label": LABEL_ORDER[int(prediction)],
                "probabilities": [float(value) for value in probability],
            }
        )
    limitations = [dict(item) for item in RESULT_LIMITATIONS]
    if dataset == "msp_podcast" and (
        set_signature.get("exclusion_contract") is not None
        or set_signature.get("duplicate_exclusion_contract") is not None
    ):
        missing_contract = set_signature.get("exclusion_contract") or {}
        duplicate_contract = set_signature.get("duplicate_exclusion_contract") or {}
        missing_counts = missing_contract.get("counts") or {}
        missing_split_counts = missing_counts.get("official_split") or {}
        duplicate_split_counts = duplicate_contract.get("excluded_split_counts") or {}
        included_count = duplicate_contract.get("final_included", missing_contract.get("final_included"))
        limitations.append(
            {
                "id": "msp_podcast_r1_10_approved_contract_subset_v1",
                "status": "contract_defined_subset",
                "excluded_missing_utterances": int(missing_contract.get("count", 0)),
                "excluded_missing_test1_utterances": int(missing_split_counts.get("Test1", 0)),
                "excluded_duplicate_utterances": int(duplicate_contract.get("count", 0)),
                "excluded_duplicate_test_utterances": int(duplicate_split_counts.get("test", 0)),
                "included_utterances": int(included_count) if included_count is not None else None,
                "missingness_assumption": "none",
                "implication": "Metrics apply to the SHA-approved missing-audio and duplicate-exclusion contracts.",
            }
        )
    if dataset == "iemocap":
        limitations.append(
            {
                "id": "iemocap_disgust_support_is_two",
                "status": "descriptive_only",
                "implication": "Do not draw general disgust-performance conclusions from this external test.",
            }
        )
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "label_order": list(LABEL_ORDER),
        "set_signature": dict(set_signature),
        "limitations": limitations,
        "metrics_4class": metrics_4class,
        "metrics_primary_3class": primary_3class,
        "predictions": prediction_rows,
    }


def evaluation_set_signature(manifest_path: str | Path, dataset: str, split: str = "test") -> dict[str, Any]:
    all_rows = load_manifest(manifest_path)
    manifest_validation = validate_manifest_records(all_rows)
    rows = [
        row
        for row in all_rows
        if row["included"] and row["dataset"] == dataset and row["split"] == split
    ]
    if not rows:
        raise ValueError(f"evaluation manifest set is empty: {dataset}/{split}")
    ids = sorted(row["utterance_id"] for row in rows)
    ids_hash = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    return {
        "dataset": dataset,
        "split": split,
        "manifest_sha256": manifest_sha256(manifest_path),
        "exclusion_contract": manifest_validation["exclusion_contracts"].get(dataset),
        "duplicate_audit": manifest_validation["duplicate_provenance"].get(dataset, {}).get("audit"),
        "duplicate_exclusion_contract": manifest_validation["duplicate_provenance"].get(dataset, {}).get(
            "exclusion_contract"
        ),
        "utterance_id_sha256": ids_hash,
        "utterance_count": len(ids),
    }


def assert_same_evaluation_sets(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    keys = (
        "dataset",
        "split",
        "manifest_sha256",
        "exclusion_contract",
        "duplicate_audit",
        "duplicate_exclusion_contract",
        "utterance_id_sha256",
        "utterance_count",
    )
    for key in keys:
        if before.get(key) != after.get(key):
            raise ValueError(f"before/after evaluation set mismatch for {key}")


def evaluate_model(model, loader, device: str | torch.device, *, dataset: str, split: str, set_signature):
    torch_device = torch.device(device)
    model.eval()
    utterance_ids: list[str] = []
    truths: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["net_input"]["feats"].to(torch_device)
            mask = batch["net_input"]["padding_mask"].to(torch_device)
            logits = model(features, mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.append(probs)
            truths.extend(int(value) for value in batch["labels"].tolist())
            utterance_ids.extend(str(value) for value in batch["utterance_ids"])
    if not probabilities:
        raise ValueError("evaluation loader is empty")
    return build_evaluation_result(
        utterance_ids,
        truths,
        np.concatenate(probabilities, axis=0),
        dataset=dataset,
        split=split,
        set_signature=set_signature,
    )


def save_evaluation_result(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    metrics_payload = {key: value for key, value in result.items() if key != "predictions"}
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    confusion_path = directory / "confusion_matrix.csv"
    matrix = result["metrics_4class"]["confusion_matrix"]
    with confusion_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["true\\predicted", *LABEL_ORDER])
        for label, row in zip(LABEL_ORDER, matrix):
            writer.writerow([label, *row])

    classes_path = directory / "class_metrics.csv"
    with classes_path.open("w", encoding="utf-8", newline="") as destination:
        fields = ("class_index", "class_label", "precision", "recall", "f1", "support")
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["metrics_4class"]["class_metrics"])

    predictions_path = directory / "predictions.csv"
    fields = [
        "utterance_id",
        "true_class_index",
        "true_label",
        "predicted_class_index",
        "predicted_label",
        *[f"probability_{label}" for label in LABEL_ORDER],
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in result["predictions"]:
            flat = {key: value for key, value in row.items() if key != "probabilities"}
            flat.update({f"probability_{label}": value for label, value in zip(LABEL_ORDER, row["probabilities"])})
            writer.writerow(flat)
    predictions_json_path = directory / "predictions.json"
    predictions_json_path.write_text(
        json.dumps(result["predictions"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": str(metrics_path),
        "confusion_matrix": str(confusion_path),
        "class_metrics": str(classes_path),
        "predictions": str(predictions_path),
        "predictions_json": str(predictions_json_path),
    }


__all__ = [
    "assert_same_evaluation_sets",
    "build_evaluation_result",
    "classification_metrics",
    "confusion_matrix",
    "evaluate_model",
    "evaluation_set_signature",
    "save_evaluation_result",
]
