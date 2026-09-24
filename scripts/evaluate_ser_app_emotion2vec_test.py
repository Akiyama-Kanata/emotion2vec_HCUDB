"""Evaluate the research-lab Gradio app's emotion2vec+ large inference on C test WAVs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import imageio_ffmpeg
import numpy as np
import torch
from funasr import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser_pipeline.contracts import OFFICIAL_TARGET_ORDER, OFFICIAL_TARGET_TO_INDEX
from ser_pipeline.evaluation import official_classification_metrics
from ser_pipeline.official import OfficialLabelSpec


APP_ROOT = Path(r"C:\Users\RD004\Documents\lab\ser_app_project_v2")
MODEL_ID = "emotion2vec/emotion2vec_plus_large"
CANONICAL_LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)
DATASET_DEFAULTS = {
    "msp_podcast": {
        "manifest": PROJECT_ROOT / "runs" / "ser_manifests" / "msp_podcast_official6_v1.jsonl",
        "audio_root": Path(r"C:\Users\RD004\Documents\lab\data\MSP_PODCAST"),
    },
    "hcudb1": {
        "manifest": PROJECT_ROOT / "runs" / "ser_manifests" / "hcudb1_official6_v1.jsonl",
        "audio_root": Path(r"C:\Users\RD004\Documents\lab\data\HCUDB1\HCUDB1"),
    },
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading large model files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file to avoid incomplete summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_test_rows(path: Path) -> list[dict[str, Any]]:
    """Load the included test rows in manifest order."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("included") is True and row.get("split") == "test":
                label = str(row.get("mapped_emotion"))
                if label not in OFFICIAL_TARGET_TO_INDEX:
                    raise ValueError(f"unexpected target label: {label!r}")
                rows.append(row)
    if not rows:
        raise ValueError(f"no included test rows: {path}")
    identifiers = [str(row["utterance_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate test utterance IDs: {path}")
    return rows


def batches(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield fixed-size manifest-order batches."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def existing_predictions(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    """Load durable prediction rows for resumable inference."""
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != dataset:
                raise ValueError(f"prediction dataset mismatch at line {line_number}: {path}")
            identifier = str(row["utterance_id"])
            if identifier in result:
                raise ValueError(f"duplicate saved prediction: {dataset}/{identifier}")
            result[identifier] = row
    return result


def configure_app_audio_loader() -> str:
    """Expose the same bundled FFmpeg executable configured at app startup."""
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    os.environ["PATH"] = str(executable.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(executable)


def model_provenance(model: AutoModel, ffmpeg_executable: str) -> dict[str, Any]:
    """Record the app, runtime, and downloaded snapshot used for inference."""
    snapshot = Path(model.model_path).resolve()
    hashes = {}
    for name in ("model.pt", "config.yaml", "tokens.txt"):
        candidate = snapshot / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        hashes[name] = {
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    app_path = APP_ROOT / "src" / "app.py"
    return {
        "app_root": str(APP_ROOT),
        "app_py_sha256": sha256_file(app_path),
        "model_id": MODEL_ID,
        "hub": "hf",
        "snapshot_path": str(snapshot),
        "snapshot_revision": snapshot.name,
        "snapshot_files": hashes,
        "funasr": importlib.metadata.version("funasr"),
        "torch": torch.__version__,
        "device": str(next(model.model.parameters()).device),
        "ffmpeg_executable": ffmpeg_executable,
        "training_performed": False,
        "inference_call": {
            "granularity": "utterance",
            "extract_embedding": False,
            "output_dir": None,
        },
    }


def prediction_record(
    dataset: str,
    source: dict[str, Any],
    result: dict[str, Any],
    expected_spec: OfficialLabelSpec | None,
) -> tuple[dict[str, Any], OfficialLabelSpec]:
    """Validate one FunASR result and convert it to the study label contract."""
    labels = tuple(str(value) for value in result["labels"])
    spec = OfficialLabelSpec(labels)
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("FunASR label order changed within the evaluation")
    probabilities = np.asarray(result["scores"], dtype=np.float64)
    if probabilities.shape != (9,) or not np.isfinite(probabilities).all():
        raise ValueError("FunASR must return nine finite scores")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise ValueError("FunASR scores must be non-negative and sum to one")
    predicted_index = int(probabilities.argmax())
    true_label = str(source["mapped_emotion"])
    true_target_index = OFFICIAL_TARGET_TO_INDEX[true_label]
    true_official_index = spec.target_indices[true_target_index]
    record = {
        "dataset": dataset,
        "utterance_id": str(source["utterance_id"]),
        "audio_relpath": str(source["audio_relpath"]),
        "original_emotion": str(source["original_emotion"]),
        "true_target_index": int(true_target_index),
        "true_target_label": true_label,
        "true_official_index": int(true_official_index),
        "true_official_label": spec.labels[true_official_index],
        "predicted_official_index": predicted_index,
        "predicted_official_label": spec.labels[predicted_index],
        "correctness": predicted_index == true_official_index,
        "probabilities9": [float(value) for value in probabilities],
    }
    return record, spec


def summarize(
    dataset: str,
    ordered: list[dict[str, Any]],
    label_spec: OfficialLabelSpec,
    expected_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Compute the experiment-C six-class metrics from app probabilities."""
    truth = [int(row["true_target_index"]) for row in ordered]
    probabilities = np.asarray([row["probabilities9"] for row in ordered], dtype=np.float64)
    metrics = official_classification_metrics(truth, probabilities, label_spec)
    return {
        "schema_version": "ser_app_emotion2vec_test_evaluation_v1",
        "dataset": dataset,
        "split": "test",
        "status": "complete" if len(ordered) == expected_count else "partial",
        "evaluated_utterances": len(ordered),
        "expected_utterances": expected_count,
        "target_label_order": list(OFFICIAL_TARGET_ORDER),
        "official_label_spec": label_spec.as_dict(),
        "decision_rule": "app_scores_argmax_official9_target6_metrics",
        "metrics_target6": metrics,
        "elapsed_seconds_this_process": elapsed_seconds,
        "training_performed": False,
    }


def evaluate_dataset(
    model: AutoModel,
    dataset: str,
    manifest: Path,
    audio_root: Path,
    output_root: Path,
    chunk_size: int,
    limit: int | None,
) -> dict[str, Any]:
    """Run resumable app inference for one dataset and write predictions and metrics."""
    all_rows = manifest_test_rows(manifest)
    selected_rows = all_rows if limit is None else all_rows[:limit]
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    saved = existing_predictions(predictions_path, dataset)
    selected_ids = {str(row["utterance_id"]) for row in selected_rows}
    unexpected = set(saved) - selected_ids
    if unexpected:
        raise ValueError(f"saved predictions are outside the selected set: {sorted(unexpected)[:3]}")

    pending = [row for row in selected_rows if str(row["utterance_id"]) not in saved]
    print(
        f"dataset={dataset} selected={len(selected_rows)} saved={len(saved)} pending={len(pending)}",
        flush=True,
    )
    label_spec: OfficialLabelSpec | None = None
    if saved:
        first = next(iter(saved.values()))
        labels = tuple(first.get("official_labels", CANONICAL_LABELS))
        label_spec = OfficialLabelSpec(labels)

    started = time.perf_counter()
    with predictions_path.open("a", encoding="utf-8", newline="\n") as handle:
        processed = len(saved)
        for chunk in batches(pending, chunk_size):
            paths = []
            for source in chunk:
                path = audio_root.joinpath(*Path(str(source["audio_relpath"])).parts)
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths.append(str(path))
            results = model.generate(
                paths,
                output_dir=None,
                granularity="utterance",
                extract_embedding=False,
                batch_size=len(paths),
            )
            if len(results) != len(chunk):
                raise ValueError("FunASR result count differs from the submitted WAV count")
            for source, result in zip(chunk, results):
                record, label_spec = prediction_record(dataset, source, result, label_spec)
                record["official_labels"] = list(label_spec.labels)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                saved[record["utterance_id"]] = record
            handle.flush()
            processed += len(chunk)
            elapsed = time.perf_counter() - started
            rate = len(pending[: max(processed - (len(selected_rows) - len(pending)), 0)]) / elapsed if elapsed else 0.0
            print(
                f"dataset={dataset} progress={processed}/{len(selected_rows)} elapsed_s={elapsed:.1f} wav_per_s={rate:.3f}",
                flush=True,
            )

    if label_spec is None:
        raise ValueError(f"no predictions available for {dataset}")
    ordered = [saved[str(row["utterance_id"])] for row in selected_rows]
    summary = summarize(
        dataset,
        ordered,
        label_spec,
        len(all_rows),
        time.perf_counter() - started,
    )
    summary["manifest"] = str(manifest.resolve())
    summary["manifest_sha256"] = sha256_file(manifest)
    summary["audio_root"] = str(audio_root.resolve())
    summary["predictions"] = str(predictions_path.resolve())
    atomic_json(output_dir / "metrics.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for smoke, partial, or complete evaluation runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("both", "msp_podcast", "hcudb1"),
        default="both",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "ser_app_emotion2vec_large_test_eval",
    )
    parser.add_argument("--msp-manifest", type=Path, default=DATASET_DEFAULTS["msp_podcast"]["manifest"])
    parser.add_argument("--msp-audio-root", type=Path, default=DATASET_DEFAULTS["msp_podcast"]["audio_root"])
    parser.add_argument("--hcudb-manifest", type=Path, default=DATASET_DEFAULTS["hcudb1"]["manifest"])
    parser.add_argument("--hcudb-audio-root", type=Path, default=DATASET_DEFAULTS["hcudb1"]["audio_root"])
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    """Load the app model once, run requested test inference, and save summaries."""
    args = parse_args()
    ffmpeg_executable = configure_app_audio_loader()
    model = AutoModel(model=MODEL_ID, hub="hf")
    model.model.requires_grad_(False).eval()
    provenance = model_provenance(model, ffmpeg_executable)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "provenance.json", provenance)

    datasets = ("msp_podcast", "hcudb1") if args.dataset == "both" else (args.dataset,)
    summaries = []
    for dataset in datasets:
        if dataset == "msp_podcast":
            manifest, audio_root = args.msp_manifest, args.msp_audio_root
        else:
            manifest, audio_root = args.hcudb_manifest, args.hcudb_audio_root
        summaries.append(
            evaluate_dataset(
                model,
                dataset,
                manifest,
                audio_root,
                args.output_root,
                args.chunk_size,
                args.limit,
            )
        )
    available_summaries = {}
    for metrics_path in args.output_root.glob("*/metrics.json"):
        item = json.loads(metrics_path.read_text(encoding="utf-8"))
        available_summaries[str(item["dataset"])] = item
    ordered_summaries = [
        available_summaries[name]
        for name in ("msp_podcast", "hcudb1")
        if name in available_summaries
    ]
    atomic_json(
        args.output_root / "summary.json",
        {
            "schema_version": "ser_app_emotion2vec_test_summary_v1",
            "status": (
                "complete"
                if set(available_summaries) == {"msp_podcast", "hcudb1"}
                and all(item["status"] == "complete" for item in ordered_summaries)
                else "partial"
            ),
            "training_performed": False,
            "provenance": str((args.output_root / "provenance.json").resolve()),
            "evaluations": ordered_summaries,
        },
    )
    for item in summaries:
        metrics = item["metrics_target6"]
        print(
            "RESULT "
            f"dataset={item['dataset']} n={item['evaluated_utterances']} "
            f"accuracy={metrics['accuracy']:.8f} uar={metrics['uar']:.8f} "
            f"macro_f1={metrics['macro_f1']:.8f} loss={metrics['loss']:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
