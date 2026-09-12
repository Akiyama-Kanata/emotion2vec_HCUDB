"""Analyze and atomically render saved official6 training histories."""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cache import _atomic_json


OFFICIAL_TRAINING_DIAGNOSTICS_SCHEMA_VERSION = "ser_official_training_diagnostics_v1"
_SCORE_NAMES = ("uar", "macro_f1", "accuracy")
_GAP_NAMES = ("loss", *_SCORE_NAMES)
_NON_TARGET_LABELS = ("neutral", "other", "unknown")


@dataclass(frozen=True)
class OfficialTrainingDiagnosticsConfig:
    """Thresholds for advisory-only official6 learning-curve diagnostics."""

    tail_epochs: int = 3
    min_score_delta: float = 0.02
    min_loss_delta: float = 0.03
    low_train_score_threshold: float = 0.50


def _validate_config(config: OfficialTrainingDiagnosticsConfig) -> None:
    if not isinstance(config, OfficialTrainingDiagnosticsConfig):
        raise ValueError("diagnostics_config must be OfficialTrainingDiagnosticsConfig")
    if isinstance(config.tail_epochs, bool) or not isinstance(config.tail_epochs, int) or config.tail_epochs < 3:
        raise ValueError("tail_epochs must be an integer of at least 3")
    for name in ("min_score_delta", "min_loss_delta", "low_train_score_threshold"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")


def _train_metrics(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metrics = row.get("train")
    if metrics is None:
        metrics = row.get("train_monitor")
    return metrics if isinstance(metrics, Mapping) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_value(metrics: Mapping[str, Any] | None, key: str) -> float | None:
    if metrics is None:
        return None
    if key == "accuracy" and "accuracy" not in metrics:
        return _finite_number(metrics.get("wa"))
    return _finite_number(metrics.get(key))


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def _below(value: float, threshold: float) -> bool:
    return value < threshold and not math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def _epoch_metrics(history: Sequence[Mapping[str, Any]], best_epoch: int | None) -> list[dict[str, Any]]:
    rows = []
    for entry in history:
        train = _train_metrics(entry)
        validation = entry.get("validation") if isinstance(entry.get("validation"), Mapping) else None
        row = {
            "epoch": entry.get("epoch"),
            "is_best": entry.get("epoch") == best_epoch,
            "optimization_train_loss": _finite_number(entry.get("train_loss")),
        }
        for key in ("loss", *_SCORE_NAMES):
            train_value = _metric_value(train, key)
            validation_value = _metric_value(validation, key)
            row[f"train_{key}"] = train_value
            row[f"validation_{key}"] = validation_value
            row[f"{key}_gap"] = (
                validation_value - train_value if key == "loss" else train_value - validation_value
            ) if train_value is not None and validation_value is not None else None
        rows.append(row)
    return rows


def analyze_official_training_history(
    history: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int | None,
    train_monitoring: Mapping[str, Any] | None,
    config: OfficialTrainingDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Return gap summaries and an advisory diagnosis without mutating training state."""
    selected = config or OfficialTrainingDiagnosticsConfig()
    _validate_config(selected)
    rows = _epoch_metrics(history, best_epoch)
    reasons: list[str] = []
    if len(rows) < selected.tail_epochs:
        reasons.append(f"fewer_than_{selected.tail_epochs}_epochs")
    if not isinstance(train_monitoring, Mapping):
        reasons.append("train_evaluation_scope_missing")
    else:
        sample_size = train_monitoring.get("sample_size")
        population_size = train_monitoring.get("population_size")
        if bool(train_monitoring.get("is_subset")) or (
            isinstance(sample_size, int) and isinstance(population_size, int) and sample_size != population_size
        ):
            reasons.append("partial_train_evaluation")
        elif not isinstance(sample_size, int) or not isinstance(population_size, int):
            reasons.append("train_evaluation_scope_missing")
    required = (
        "optimization_train_loss", "train_loss", "validation_loss",
        "train_uar", "validation_uar", "train_macro_f1", "validation_macro_f1",
        "train_accuracy", "validation_accuracy",
    )
    if any(row.get(key) is None for row in rows for key in required):
        reasons.append("missing_or_non_finite_metrics")
    epochs = [row.get("epoch") for row in rows]
    if any(isinstance(epoch, bool) or not isinstance(epoch, int) for epoch in epochs):
        reasons.append("invalid_epoch_values")
    elif epochs != list(range(1, len(rows) + 1)):
        reasons.append("non_contiguous_history")

    gap_summary: dict[str, Any] = {"final": None, "best_epoch": None, "maximum": None}
    if rows:
        gap_summary["final"] = {key: rows[-1][f"{key}_gap"] for key in _GAP_NAMES}
        best = next((row for row in rows if row["epoch"] == best_epoch), None)
        if best is not None:
            gap_summary["best_epoch"] = {key: best[f"{key}_gap"] for key in _GAP_NAMES}
        gap_summary["maximum"] = {
            key: max(
                (row[f"{key}_gap"] for row in rows if row[f"{key}_gap"] is not None),
                default=None,
            )
            for key in _GAP_NAMES
        }

    evidence: dict[str, Any] = {}
    status = "indeterminate"
    if not reasons:
        tail = rows[-selected.tail_epochs:]
        first, final = tail[0], tail[-1]
        train_loss_decrease = first["train_loss"] - final["train_loss"]
        validation_loss_increase = final["validation_loss"] - first["validation_loss"]
        train_loss_improvement = train_loss_decrease
        score_improvements = {
            key: final[f"train_{key}"] - first[f"train_{key}"] for key in ("uar", "macro_f1")
        }
        gap_expansions = {
            key: final[f"{key}_gap"] - first[f"{key}_gap"] for key in ("uar", "macro_f1")
        }
        overfitting = (
            _at_least(train_loss_decrease, selected.min_loss_delta)
            and _at_least(validation_loss_increase, selected.min_loss_delta)
            and any(_at_least(value, selected.min_score_delta) for value in gap_expansions.values())
        )
        underfitting = (
            _below(final["train_uar"], selected.low_train_score_threshold)
            and _below(final["train_macro_f1"], selected.low_train_score_threshold)
            and all(_below(value, selected.min_score_delta) for value in score_improvements.values())
            and _below(train_loss_improvement, selected.min_loss_delta)
        )
        status = "overfitting" if overfitting else "underfitting" if underfitting else "good"
        evidence = {
            "tail_epoch_range": [first["epoch"], final["epoch"]],
            "train_loss_decrease": train_loss_decrease,
            "validation_loss_increase": validation_loss_increase,
            "train_score_improvements": score_improvements,
            "score_gap_expansions": gap_expansions,
            "final_train_scores": {key: final[f"train_{key}"] for key in ("uar", "macro_f1")},
            "overfitting_conditions_met": overfitting,
            "underfitting_conditions_met": underfitting,
        }
    return {
        "judgement": {
            "status": status,
            "reasons": reasons,
            "evidence": evidence,
            "advisory_only": True,
            "affects_training_or_checkpoint_selection": False,
        },
        "gap_definitions": {
            "scores": "train - validation",
            "loss": "validation - train",
        },
        "gap_summary": gap_summary,
        "epoch_metrics": rows,
    }


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def _class_rows(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in history:
        for split, metrics in (("train", _train_metrics(entry)), ("validation", entry.get("validation"))):
            if not isinstance(metrics, Mapping):
                continue
            for item in metrics.get("class_metrics", []):
                rows.append({"epoch": entry.get("epoch"), "split": split, **dict(item)})
    return rows


def _non_target_rows(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in history:
        for split, metrics in (("train", _train_metrics(entry)), ("validation", entry.get("validation"))):
            if not isinstance(metrics, Mapping):
                continue
            values = metrics.get("non_target_predictions")
            if not isinstance(values, Mapping):
                continue
            counts = values.get("counts", {})
            rates = values.get("rates", {})
            for label in _NON_TARGET_LABELS:
                rows.append({
                    "epoch": entry.get("epoch"), "split": split, "official_label": label,
                    "count": counts.get(label), "rate": rates.get(label),
                })
            rows.append({
                "epoch": entry.get("epoch"), "split": split, "official_label": "all_non_target",
                "count": values.get("total"), "rate": values.get("rate"),
            })
    return rows


def _atomic_learning_curves(rows: Sequence[Mapping[str, Any]], best_epoch: int | None, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    epochs = [row["epoch"] for row in rows]
    for axis, metric, title in zip(
        axes.flat,
        ("loss", "uar", "macro_f1", "accuracy"),
        ("Comparison loss", "UAR", "Macro F1", "Accuracy"),
    ):
        axis.plot(epochs, [row[f"train_{metric}"] for row in rows], marker="o", label="train")
        axis.plot(epochs, [row[f"validation_{metric}"] for row in rows], marker="o", label="validation")
        if best_epoch is not None:
            axis.axvline(best_epoch, color="#64748b", linestyle="--", label="best epoch")
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    figure.savefig(partial, format="png", dpi=150)
    plt.close(figure)
    partial.replace(path)


def write_official_training_diagnostics(
    history: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    run_id: str,
    seed: int,
    best_epoch: int | None,
    train_monitoring: Mapping[str, Any] | None,
    config: OfficialTrainingDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Atomically replace one run's CSV, PNG, and JSON diagnostic artifacts."""
    selected = config or OfficialTrainingDiagnosticsConfig()
    analysis = analyze_official_training_history(
        history, best_epoch=best_epoch, train_monitoring=train_monitoring, config=selected,
    )
    directory = Path(output_dir).resolve() / "diagnostics"
    paths = {
        "epoch_metrics_csv": directory / "epoch_metrics.csv",
        "class_metrics_by_epoch_csv": directory / "class_metrics_by_epoch.csv",
        "non_target_predictions_by_epoch_csv": directory / "non_target_predictions_by_epoch.csv",
        "learning_curves_png": directory / "learning_curves.png",
        "training_diagnostics_json": directory / "training_diagnostics.json",
    }
    epoch_fields = ["epoch", "is_best", "optimization_train_loss"]
    for key in ("loss", *_SCORE_NAMES):
        epoch_fields.extend((f"train_{key}", f"validation_{key}", f"{key}_gap"))
    _atomic_csv(paths["epoch_metrics_csv"], epoch_fields, analysis["epoch_metrics"])
    class_rows = _class_rows(history)
    _atomic_csv(
        paths["class_metrics_by_epoch_csv"],
        ("epoch", "split", "target_index", "target_label", "official_index", "precision", "recall", "f1", "support"),
        class_rows,
    )
    non_target_rows = _non_target_rows(history)
    _atomic_csv(
        paths["non_target_predictions_by_epoch_csv"],
        ("epoch", "split", "official_label", "count", "rate"),
        non_target_rows,
    )
    _atomic_learning_curves(analysis["epoch_metrics"], best_epoch, paths["learning_curves_png"])
    epochs = [row.get("epoch") for row in history]
    payload = {
        "diagnostics_schema_version": OFFICIAL_TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
        "run_id": str(run_id),
        "seed": int(seed),
        "history_range": {
            "first_epoch": epochs[0] if epochs else None,
            "last_epoch": epochs[-1] if epochs else None,
            "epoch_count": len(epochs),
            "best_epoch": best_epoch,
        },
        "thresholds": asdict(selected),
        "train_evaluation": dict(train_monitoring) if isinstance(train_monitoring, Mapping) else None,
        "judgement": analysis["judgement"],
        "gap_definitions": analysis["gap_definitions"],
        "gap_summary": analysis["gap_summary"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _atomic_json(payload, paths["training_diagnostics_json"])
    return payload


def aggregate_official_training_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], *, requested_seed_count: int = 3,
) -> dict[str, Any]:
    """Aggregate final and selected-best signed gaps with sample standard deviations."""
    if requested_seed_count <= 0:
        raise ValueError("requested_seed_count must be positive")

    def aggregate_point(point: str) -> dict[str, Any]:
        result = {}
        for metric in _GAP_NAMES:
            values = []
            for payload in diagnostics:
                gaps = payload.get("gap_summary", {}).get(point)
                value = gaps.get(metric) if isinstance(gaps, Mapping) else None
                number = _finite_number(value)
                if number is not None:
                    values.append(number)
            result[metric] = {
                "mean": statistics.fmean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "count": len(values),
            }
        return result

    observed_counts = Counter(
        str(payload.get("judgement", {}).get("status", "indeterminate")) for payload in diagnostics
    )
    counts = {
        status: observed_counts.get(status, 0)
        for status in ("overfitting", "underfitting", "good", "indeterminate")
    }
    return {
        "status": "complete" if len(diagnostics) == requested_seed_count else "partial",
        "requested_seed_count": requested_seed_count,
        "completed_seed_count": len(diagnostics),
        "final_gaps": aggregate_point("final"),
        "best_epoch_gaps": aggregate_point("best_epoch"),
        "diagnosis_counts": counts,
    }


__all__ = [
    "OFFICIAL_TRAINING_DIAGNOSTICS_SCHEMA_VERSION",
    "OfficialTrainingDiagnosticsConfig",
    "aggregate_official_training_diagnostics",
    "analyze_official_training_history",
    "write_official_training_diagnostics",
]
