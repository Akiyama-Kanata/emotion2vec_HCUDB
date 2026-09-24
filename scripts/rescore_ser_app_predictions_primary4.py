"""Rescore saved ser_app emotion2vec+ large predictions under the primary four-class contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "runs"
    / "ser_app_emotion2vec_large_test_eval"
    / "msp_podcast"
    / "predictions.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("metrics_primary4.json")
LABEL_ORDER = ("angry", "happy", "sad", "disgusted")
OFFICIAL_INDICES = (0, 3, 6, 1)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load saved JSONL predictions in file order."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Filter primary-four truth and score argmax over the same four outputs."""
    selected = [row for row in rows if row["true_target_label"] in LABEL_ORDER]
    if not selected:
        raise ValueError("no primary-four prediction rows found")

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    truth = np.asarray(
        [label_to_index[row["true_target_label"]] for row in selected],
        dtype=np.int64,
    )
    probabilities9 = np.asarray(
        [row["probabilities9"] for row in selected],
        dtype=np.float64,
    )
    probabilities4 = probabilities9[:, OFFICIAL_INDICES]
    probabilities4 /= probabilities4.sum(axis=1, keepdims=True)
    prediction = probabilities4.argmax(axis=1)

    matrix = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    class_metrics = []
    recalls = []
    f1_values = []
    for index, label in enumerate(LABEL_ORDER):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics.append(
            {
                "class_index": index,
                "class_label": label,
                "official_index": OFFICIAL_INDICES[index],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        recalls.append(recall)
        f1_values.append(f1)

    loss = -np.log(
        np.clip(probabilities4[np.arange(len(truth)), truth], 1e-12, 1.0)
    ).mean()
    excluded_counts = {
        label: sum(row["true_target_label"] == label for row in rows)
        for label in ("fearful", "surprised")
    }
    return {
        "schema_version": "ser_app_emotion2vec_primary4_rescore_v1",
        "dataset": "msp_podcast",
        "split": "test",
        "decision_rule": "filter_primary4_truth_then_argmax_primary4_probabilities",
        "training_performed": False,
        "inference_performed": False,
        "source_utterances": len(rows),
        "evaluated_utterances": len(selected),
        "excluded_truth_counts": excluded_counts,
        "label_order": list(LABEL_ORDER),
        "official_indices": list(OFFICIAL_INDICES),
        "accuracy": float(np.mean(truth == prediction)),
        "uar": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "loss": float(loss),
        "class_metrics": class_metrics,
        "confusion_matrix": matrix.tolist(),
    }


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the deterministic post-hoc rescore and save its provenance."""
    args = parse_args()
    rows = load_rows(args.input)
    result = score(rows)
    result["source_predictions"] = str(args.input.resolve())
    result["source_predictions_sha256"] = sha256_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"RESULT n={result['evaluated_utterances']} "
        f"accuracy={result['accuracy']:.8f} uar={result['uar']:.8f} "
        f"macro_f1={result['macro_f1']:.8f} loss={result['loss']:.8f}"
    )
    print(f"OUTPUT {args.output.resolve()}")


if __name__ == "__main__":
    main()
