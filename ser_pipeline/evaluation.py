"""Four-class metrics, IEMOCAP three-class summary, and result persistence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .contracts import (
    LABEL_ORDER,
    OFFICIAL_EVALUATION_METHOD,
    OFFICIAL_RESULT_SCHEMA_VERSION,
    OFFICIAL_TARGET_ORDER,
    PRIMARY_EVALUATION_METHOD,
    RESULT_LIMITATIONS,
    RESULT_SCHEMA_VERSION,
    label_profile_for_mapping_version,
)
from .manifest import load_manifest, manifest_sha256, validate_manifest_records
from .timing import measure, timed_batches


def select_primary_logits(logits, label_spec=None):
    """Use common-order A/B logits or dictionary-resolved C/D columns."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, classes]")
    if label_spec is None:
        if logits.shape[1] != len(LABEL_ORDER):
            raise ValueError("nine logits require an official label specification")
        return logits
    if logits.shape[1] != 9:
        raise ValueError("official model must return nine logits")
    return logits[:, list(label_spec.primary_indices)]


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
    """Return UAR, macro F1, accuracy/WA and optional unweighted comparison loss.

    UAR and macro F1 average recall and F1 over ``reported_classes`` (all four
    classes by default); absent classes and undefined precision/recall/F1 use
    zero. Accuracy and WA are the same fraction of correctly predicted utterances.
    Loss is ``-mean(log(clip(p_true, 1e-12, 1)))`` over all input utterances,
    with equal utterance weights, independent of batching or training weights.
    This probability-based calculation intentionally differs from the saved
    mean of optimization batch losses, and is not replaced by a logits formula.
    """
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


def official_confusion_matrix(
    y_true_target6: Sequence[int],
    y_pred_official: Sequence[int],
) -> np.ndarray:
    """Build the C/D matrix with target6 truth rows and official prediction columns."""
    truth = np.asarray(y_true_target6, dtype=np.int64)
    prediction = np.asarray(y_pred_official, dtype=np.int64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("official truth and prediction must be aligned non-empty vectors")
    if np.any(truth < 0) or np.any(truth >= len(OFFICIAL_TARGET_ORDER)):
        raise ValueError("official truth must use contiguous target6 indices")
    if np.any(prediction < 0) or np.any(prediction >= 9):
        raise ValueError("official predictions must be indices 0 through 8")
    matrix = np.zeros((len(OFFICIAL_TARGET_ORDER), 9), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def official_classification_metrics(
    y_true_target6: Sequence[int],
    probabilities9: np.ndarray,
    label_spec,
) -> dict[str, Any]:
    """Score six-class truth under an unchanged nine-way official decision rule."""
    truth = np.asarray(y_true_target6, dtype=np.int64)
    probs = np.asarray(probabilities9, dtype=np.float64)
    if probs.shape != (len(truth), 9) or len(truth) == 0:
        raise ValueError("official probabilities must have shape [samples, 9]")
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(1), 1, atol=1e-6):
        raise ValueError("official probabilities must be finite, non-negative, and sum to one")
    target_indices = np.asarray(label_spec.target_indices, dtype=np.int64)
    if tuple(label_spec.target_names) != OFFICIAL_TARGET_ORDER or tuple(target_indices) != (0, 1, 2, 3, 6, 7):
        raise ValueError("official target label contract mismatch")
    if np.any(truth < 0) or np.any(truth >= len(target_indices)):
        raise ValueError("official truth must use contiguous target6 indices")
    true_official = target_indices[truth]
    prediction = probs.argmax(axis=1).astype(np.int64)
    matrix = official_confusion_matrix(truth, prediction)
    class_rows = []
    recalls = []
    f1_values = []
    for target_index, (label, official_index) in enumerate(zip(OFFICIAL_TARGET_ORDER, target_indices)):
        true_positive = int(matrix[target_index, official_index])
        support = int(matrix[target_index].sum())
        predicted = int(matrix[:, official_index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append({
            "target_index": target_index,
            "target_label": label,
            "official_index": int(official_index),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        })
        recalls.append(recall)
        f1_values.append(f1)
    fixed_counts = {
        label_spec.labels[index]: int(np.count_nonzero(prediction == index))
        for index in label_spec.fixed_indices
    }
    total = len(truth)
    return {
        "accuracy": float(np.mean(prediction == true_official)),
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "loss": float(-np.log(np.clip(probs[np.arange(total), true_official], 1e-12, 1.0)).mean()),
        "target_names": list(OFFICIAL_TARGET_ORDER),
        "target_official_indices": target_indices.tolist(),
        "class_metrics": class_rows,
        "confusion_matrix_6x9": matrix.tolist(),
        "non_target_predictions": {
            "counts": fixed_counts,
            "rates": {label: float(count / total) for label, count in fixed_counts.items()},
            "total": int(sum(fixed_counts.values())),
            "rate": float(sum(fixed_counts.values()) / total),
        },
    }


def build_official_evaluation_result(
    utterance_ids: Sequence[str],
    y_true_target6: Sequence[int],
    logits9: np.ndarray,
    *,
    label_spec,
    dataset: str,
    split: str,
    set_signature: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    logits = np.asarray(logits9, dtype=np.float64)
    if logits.shape != (len(utterance_ids), 9) or len(y_true_target6) != len(utterance_ids):
        raise ValueError("official evaluation ids, labels, and logits must align")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    metrics = official_classification_metrics(y_true_target6, probabilities, label_spec)
    predicted = probabilities.argmax(axis=1)
    target_indices = np.asarray(label_spec.target_indices, dtype=np.int64)
    prediction_rows = []
    for identifier, truth, prediction, row_logits, row_probabilities in zip(
        utterance_ids, y_true_target6, predicted, logits, probabilities
    ):
        source = source_rows[str(identifier)]
        true_official = int(target_indices[int(truth)])
        prediction_rows.append({
            "utterance_id": str(identifier),
            "true_target_index": int(truth),
            "true_target_label": OFFICIAL_TARGET_ORDER[int(truth)],
            "true_official_index": true_official,
            "true_official_label": label_spec.labels[true_official],
            "predicted_official_index": int(prediction),
            "predicted_official_label": label_spec.labels[int(prediction)],
            "correctness": bool(int(prediction) == true_official),
            "original_emotion": str(source["original_emotion"]),
            "logits9": [float(value) for value in row_logits],
            "probabilities9": [float(value) for value in row_probabilities],
        })
    limitations = [dict(item) for item in RESULT_LIMITATIONS]
    if dataset == "hcudb1":
        limitations.append({
            "id": "hcudb_dislike_to_disgust",
            "approximate_mapping": True,
            "implication": "嫌い → disgusted is a research mapping assumption.",
        })
    return {
        "result_schema_version": OFFICIAL_RESULT_SCHEMA_VERSION,
        "evaluation_method": OFFICIAL_EVALUATION_METHOD,
        "decision_rule": "softmax_official9_then_argmax_official9",
        "dataset": dataset,
        "split": split,
        "target_label_order": list(OFFICIAL_TARGET_ORDER),
        "official_label_spec": label_spec.as_dict(),
        "set_signature": dict(set_signature),
        "limitations": limitations,
        "metrics_target6": metrics,
        "predictions": prediction_rows,
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
    if dataset == "hcudb1":
        limitations.append({"id": "hcudb_dislike_to_disgust", "approximate_mapping": True,
                            "implication": "嫌い → disgust is a research mapping assumption."})
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_method": PRIMARY_EVALUATION_METHOD,
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
    profiles = {label_profile_for_mapping_version(str(row["mapping_version"])) for row in rows}
    if len(profiles) != 1:
        raise ValueError("evaluation set mixes label profiles")
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
        "label_profile": next(iter(profiles)),
        "mapping_versions": sorted({str(row["mapping_version"]) for row in rows}),
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
        "label_profile",
        "mapping_versions",
    )
    for key in keys:
        if before.get(key) != after.get(key):
            raise ValueError(f"before/after evaluation set mismatch for {key}")


def assert_comparable_results(before, after):
    """Gate comparisons by exact cohorts, labels, and decision rule."""
    assert_same_evaluation_sets(before["set_signature"], after["set_signature"])
    if before.get("result_schema_version") == OFFICIAL_RESULT_SCHEMA_VERSION or after.get("result_schema_version") == OFFICIAL_RESULT_SCHEMA_VERSION:
        if (before.get("result_schema_version") != OFFICIAL_RESULT_SCHEMA_VERSION
                or after.get("result_schema_version") != OFFICIAL_RESULT_SCHEMA_VERSION
                or before.get("target_label_order") != list(OFFICIAL_TARGET_ORDER)
                or after.get("target_label_order") != list(OFFICIAL_TARGET_ORDER)
                or before.get("official_label_spec") != after.get("official_label_spec")
                or before.get("evaluation_method") != OFFICIAL_EVALUATION_METHOD
                or after.get("evaluation_method") != OFFICIAL_EVALUATION_METHOD):
            raise ValueError("comparison requires identical official9/target6 contracts")
        left = {row["utterance_id"]: row["true_official_index"] for row in before["predictions"]}
        right = {row["utterance_id"]: row["true_official_index"] for row in after["predictions"]}
        if left != right or len(left) != len(before["predictions"]) or len(right) != len(after["predictions"]):
            raise ValueError("comparison utterance IDs/official labels mismatch")
        return
    if (before.get("label_order") != list(LABEL_ORDER) or after.get("label_order") != list(LABEL_ORDER)
            or before.get("evaluation_method") != PRIMARY_EVALUATION_METHOD
            or after.get("evaluation_method") != PRIMARY_EVALUATION_METHOD):
        raise ValueError("comparison requires identical labels and explicit four-way evaluation method")
    left = {row["utterance_id"]: row["true_class_index"] for row in before["predictions"]}
    right = {row["utterance_id"]: row["true_class_index"] for row in after["predictions"]}
    if left != right or len(left) != len(before["predictions"]) or len(right) != len(after["predictions"]):
        raise ValueError("comparison utterance IDs/labels mismatch")


def evaluate_model(model, loader, device: str | torch.device, *, dataset: str, split: str, set_signature, timings=None):
    torch_device = torch.device(device)
    model.eval()
    utterance_ids: list[str] = []
    truths: list[int] = []
    probabilities: list[np.ndarray] = []
    official_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in timed_batches(loader, timings):
            with measure(timings, "compute_seconds", torch_device):
                features = batch["net_input"]["feats"].to(torch_device)
                mask = batch["net_input"]["padding_mask"].to(torch_device)
                logits = model(features, mask)
                label_spec = getattr(model, "label_spec", None)
                if label_spec is not None:
                    if logits.shape[1] != 9:
                        raise ValueError("official model must return nine logits")
                    official_logits.append(logits.cpu().numpy())
                else:
                    primary = select_primary_logits(logits, None)
                    probabilities.append(torch.softmax(primary, dim=-1).cpu().numpy())
                truths.extend(int(value) for value in batch["labels"].tolist())
                utterance_ids.extend(str(value) for value in batch["utterance_ids"])
    if not probabilities and not official_logits:
        raise ValueError("evaluation loader is empty")
    with measure(timings, "result_build_seconds"):
        if official_logits:
            dataset_object = getattr(loader, "dataset", None)
            store = getattr(dataset_object, "store", None)
            if store is None:
                raise ValueError("official evaluation requires manifest source rows")
            source_rows = {identifier: store.records[identifier] for identifier in utterance_ids}
            return build_official_evaluation_result(
                utterance_ids,
                truths,
                np.concatenate(official_logits, axis=0),
                label_spec=model.label_spec,
                dataset=dataset,
                split=split,
                set_signature=set_signature,
                source_rows=source_rows,
            )
        return build_evaluation_result(
            utterance_ids, truths, np.concatenate(probabilities, axis=0),
            dataset=dataset, split=split, set_signature=set_signature,
        )


def save_evaluation_result(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    metrics_payload = {key: value for key, value in result.items() if key != "predictions"}
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    official = result.get("result_schema_version") == OFFICIAL_RESULT_SCHEMA_VERSION
    confusion_path = directory / "confusion_matrix.csv"
    metrics = result["metrics_target6"] if official else result["metrics_4class"]
    matrix = metrics["confusion_matrix_6x9"] if official else metrics["confusion_matrix"]
    with confusion_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        predicted_labels = result["official_label_spec"]["labels"] if official else LABEL_ORDER
        true_labels = OFFICIAL_TARGET_ORDER if official else LABEL_ORDER
        writer.writerow(["true\\predicted", *predicted_labels])
        for label, row in zip(true_labels, matrix):
            writer.writerow([label, *row])

    classes_path = directory / "class_metrics.csv"
    with classes_path.open("w", encoding="utf-8", newline="") as destination:
        fields = (
            ("target_index", "target_label", "official_index", "precision", "recall", "f1", "support")
            if official else ("class_index", "class_label", "precision", "recall", "f1", "support")
        )
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["class_metrics"])

    predictions_path = directory / "predictions.csv"
    official_labels = result.get("official_label_spec", {}).get("labels", [])
    fields = ([
        "utterance_id", "true_target_index", "true_target_label", "true_official_index",
        "true_official_label", "predicted_official_index", "predicted_official_label",
        "correctness", "original_emotion",
        *[f"logit9_{label}" for label in official_labels],
        *[f"probability9_{label}" for label in official_labels],
    ] if official else [
        "utterance_id", "true_class_index", "true_label", "predicted_class_index", "predicted_label",
        *[f"probability_{label}" for label in LABEL_ORDER],
    ])
    with predictions_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in result["predictions"]:
            flat = {key: value for key, value in row.items() if key not in {"probabilities", "logits9", "probabilities9"}}
            if official:
                for key, prefix in (("logits9", "logit9"), ("probabilities9", "probability9")):
                    flat.update({f"{prefix}_{label}": value for label, value in zip(official_labels, row[key])})
            else:
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
    "assert_comparable_results",
    "assert_same_evaluation_sets",
    "build_evaluation_result",
    "build_official_evaluation_result",
    "classification_metrics",
    "confusion_matrix",
    "evaluate_model",
    "evaluation_set_signature",
    "official_classification_metrics",
    "official_confusion_matrix",
    "save_evaluation_result",
    "select_primary_logits",
]
