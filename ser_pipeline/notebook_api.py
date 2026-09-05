"""Small orchestration helpers used by the two study notebooks."""

from __future__ import annotations

import hashlib
import base64
import html
import io
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .audio import sha256_file
from .cache import CacheIndexEntry, _atomic_json, _write_index, validate_cache
from .contracts import (
    CACHE_SCHEMA_VERSION,
    EXTRACTION_CODE_VERSION,
    FEATURE_LAYER,
    LABEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    load_mapping_config,
)
from .manifest import manifest_sha256, write_manifest
from .splits import IEMOCAP_SPLIT_VERSION, MSP_SPLIT_VERSION, load_hcudb_split
from .study import DatasetArtifacts, EVALUATION_DATASETS, STUDY_SEEDS, run_transfer_study
from .training import TrainingConfig


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "label_order": list(LABEL_ORDER),
        "feature_layer": FEATURE_LAYER,
    }


def _history_rows(training):
    recorded = {int(row["epoch"]): row for row in (training.get("history") or [])
                if isinstance(row, dict) and row.get("epoch") is not None}
    if not recorded:
        return []
    # An absent epoch gets missing plot/table values, never an interpolated line.
    return [recorded.get(epoch, {"epoch": epoch}) for epoch in range(min(recorded), max(recorded) + 1)]


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _history_values(history, split, key):
    values = []
    for row in history:
        if split == "train":
            metrics = row.get("train_monitor") if "train_monitor" in row else row.get("train")
            metrics = metrics or {}
        else:
            metrics = (row.get(split) or {}) if split else row
        value = metrics.get(key)
        if key == "wa" and value is None:
            value = metrics.get("accuracy")
        values.append(_number(value))
    return values


def _train_curve_label(training):
    monitoring = training.get("train_monitoring")
    if not isinstance(monitoring, dict):
        return "train"
    sample_size = monitoring.get("sample_size")
    if monitoring.get("is_subset"):
        return f"train（固定{sample_size:,}件・参考）" if isinstance(sample_size, int) else "train（固定部分集合・参考）"
    return "train（全件・参考）"


def _finish_history_plot(figure, output_path, show):
    import matplotlib.pyplot as plt

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(figure)  # Prevent duplicate automatic notebook output.
    return figure


def plot_training_scores(training: dict[str, Any], *, output_path: str | Path | None = None, show: bool = True):
    """Show epoch scores in the training cell; absent historical train scores stay absent."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    history = _history_rows(training)
    if not history:
        print("scoreの履歴がありません。")
        return None
    epochs = [int(entry["epoch"]) for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True, layout="constrained")
    for axis, (key, label) in zip(axes, (("uar", "UAR (primary)"), ("macro_f1", "Macro F1"), ("wa", "Accuracy (reference)"))):
        for split, color in (("train", "#2563eb"), ("validation", "#ea580c")):
            values = _history_values(history, split, key)
            if any(np.isfinite(value) for value in values):
                label_name = _train_curve_label(training) if split == "train" else split
                axis.plot(epochs, values, label=label_name, marker="o", markersize=4, color=color)
        if np.isfinite(_number(training.get("best_epoch"))):
            axis.axvline(training["best_epoch"], color="#64748b", linestyle="--", label="Best validation epoch")
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, 1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Score")
    figure.suptitle(f"{training.get('dataset', '')} / seed {training.get('seed', '')}")
    return _finish_history_plot(figure, output_path, show)


def plot_training_losses(training: dict[str, Any], *, output_path: str | Path | None = None, show: bool = True):
    """Plot saved comparison losses and the separate optimization batch-loss mean."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    history = _history_rows(training)
    if not history:
        return None
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for split, color in (("train", "#2563eb"), ("validation", "#ea580c")):
        values = _history_values(history, split, "loss")
        if any(np.isfinite(values)):
            label_name = _train_curve_label(training) if split == "train" else split
            axes[0].plot(epochs, values, label=label_name, marker="o", markersize=4, color=color)
    values = _history_values(history, None, "train_loss")
    if any(np.isfinite(values)):
        axes[1].plot(epochs, values, label="train_loss", marker="o", markersize=4, color="#2563eb")
    weighting = (training.get("loss_config") or training.get("config") or {}).get("class_weighting", "unrecorded")
    axes[0].set_title("Comparison loss (unweighted utterance mean)")
    axes[1].set_title(f"Optimization loss (class_weighting={weighting})")
    for axis in axes:
        if np.isfinite(_number(training.get("best_epoch"))):
            axis.axvline(training["best_epoch"], color="#64748b", linestyle="--", label="Best validation epoch")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
    return _finish_history_plot(figure, output_path, show)


def _figure_html(figure, alt):
    if figure is None:
        return ""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" style="max-width:100%;height:auto" src="data:image/png;base64,{encoded}">'


def _format_recorded(value):
    number = _number(value)
    return f"{number:.4f}" if np.isfinite(number) else "未記録"


def _best_training_metrics_html(training):
    metrics = training.get("best_training_metrics") or {}
    if not metrics:
        return "<p>bestモデルのtrain全件結果：未記録</p>"
    support = sum(int(row.get("support", 0)) for row in metrics.get("class_metrics") or [])
    overall = ''.join(
        f'<td>{_format_recorded(metrics.get(name))}</td>'
        for name in ("uar", "macro_f1", "accuracy", "loss")
    )
    class_rows = []
    for row in metrics.get("class_metrics") or []:
        label = html.escape(str(row.get("class_label", "未記録")))
        values = ''.join(f'<td>{_format_recorded(row.get(name))}</td>' for name in ("precision", "recall", "f1"))
        class_rows.append(f'<tr><td>{label}</td>{values}<td>{html.escape(str(row.get("support", "未記録")))}</td></tr>')
    class_table = (
        '<table><thead><tr><th>感情</th><th>precision</th><th>recall</th><th>F1</th><th>件数</th></tr></thead><tbody>'
        + ''.join(class_rows) + '</tbody></table>'
    ) if class_rows else ''
    return (
        '<details><summary>bestモデルのtrain全件結果（正式）</summary>'
        '<p>固定監視部分集合ではなく、best状態をtrain全件で評価した結果です。</p>'
        '<table><thead><tr><th>件数</th><th>UAR</th><th>macro F1</th><th>accuracy</th><th>比較用loss</th></tr></thead><tbody>'
        f'<tr><td>{support}</td>{overall}</tr></tbody></table>{class_table}</details>'
    )


def _best_validation_class_metrics_html(training):
    rows = []
    metrics = training.get("best_validation_metrics") or {}
    for row in metrics.get("class_metrics") or []:
        label = html.escape(str(row.get("class_label", "未記録")))
        values = ''.join(f'<td>{_format_recorded(row.get(name))}</td>' for name in ("precision", "recall", "f1"))
        rows.append(f'<tr><td>{label}</td>{values}<td>{html.escape(str(row.get("support", "未記録")))}</td></tr>')
    if not rows:
        return ""
    return ('<details><summary>best epochのvalidationクラス別指標</summary>'
            '<table><thead><tr><th>感情</th><th>precision</th><th>recall</th><th>F1</th><th>件数</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></details>')


def _timing_html(training):
    timing = training.get("timings") or {}
    epochs = timing.get("epochs") or []
    if not epochs:
        return "<p>train評価時間：未記録</p>"
    rows = []
    for row in epochs:
        recorded_seconds = row.get("train_monitor_evaluation_seconds")
        if recorded_seconds is None:
            recorded_seconds = row.get("train_evaluation_seconds")
        seconds, total = _number(recorded_seconds), _number(row.get("total_seconds"))
        inner = row.get("train_monitor") or row.get("train_evaluation") or {}
        ratio = seconds / total if total > 0 else None
        values = [seconds, inner.get("batch_prepare_seconds"), inner.get("compute_seconds"), inner.get("metrics_seconds"), total, ratio]
        rows.append(f'<tr><td>{html.escape(str(row.get("epoch", "未記録")))}</td>' + ''.join(f'<td>{_format_recorded(value)}</td>' for value in values) + '</tr>')
    recorded = [
        _number(row.get("train_monitor_evaluation_seconds", row.get("train_evaluation_seconds")))
        for row in epochs
    ]
    complete = all(np.isfinite(recorded))
    aggregate = _format_recorded(sum(recorded)) if complete else "未記録（一部欠損）"
    later = _format_recorded(np.mean(recorded[1:])) if complete and len(recorded) > 1 else "未記録"
    return (
        '<details><summary>train監視評価の処理時間</summary>'
        '<p>単位：秒。割合はtrain監視評価時間 / epoch総時間です。新旧の速度差ではありません。</p>'
        '<table><thead><tr><th>epoch</th><th>train監視評価</th><th>バッチ準備</th><th>計算</th><th>指標集計</th><th>epoch総時間</th><th>割合</th></tr></thead><tbody>'
        + ''.join(rows) + f'</tbody></table><p>{len(epochs)} epoch合計：{aggregate}秒 / '
        f'初回：{_format_recorded(recorded[0])}秒 / 後続epoch平均：{later}秒</p>'
        f'<p>bestモデルのtrain全件評価：{_format_recorded(timing.get("best_train_evaluation_seconds"))}秒'
        f'（監視が全件の場合の再利用：{html.escape(str(timing.get("best_train_evaluation_reused_from_monitor", "未記録"))) }）</p></details>'
    )


def display_training_history(training: dict[str, Any], *, save_plots: bool = False, display_output: bool = True):
    """Render only saved history as static HTML; plotting never loads/evaluates a model.

    Set save_plots only for a new run. Replaying old summaries leaves all original
    artifacts untouched. The returned HTML is useful for export and output tests.
    """
    from IPython.display import HTML, display

    history = _history_rows(training)
    title = html.escape(f"{training.get('dataset', '')} / seed {training.get('seed', '')}")
    weighting = (training.get("loss_config") or training.get("config") or {}).get("class_weighting", "未記録")
    train_curve_label = html.escape(_train_curve_label(training))
    if not history:
        result = HTML(f"<h4>{title}</h4><p>score・lossの履歴：未記録</p>")
    else:
        checkpoint = Path(training["best_checkpoint"]) if save_plots else None
        scores = plot_training_scores(training, output_path=checkpoint.with_suffix('.scores.png') if checkpoint else None, show=False)
        losses = plot_training_losses(training, output_path=checkpoint.with_suffix('.losses.png') if checkpoint else None, show=False)
        table = []
        train_loss = _history_values(history, "train", "loss")
        val_loss = _history_values(history, "validation", "loss")
        optim_loss = _history_values(history, None, "train_loss")
        for row, train, validation, optimization in zip(history, train_loss, val_loss, optim_loss):
            best = "★" if row["epoch"] == training.get("best_epoch") else ""
            table.append(f'<tr><td>{int(row["epoch"])}</td><td>{_format_recorded(train)}</td><td>{_format_recorded(validation)}</td><td>{_format_recorded(optimization)}</td><td>{best}</td></tr>')
        result = HTML(
            f'<section><h4>{title} / class_weighting={html.escape(str(weighting))}</h4>'
            f'<p>UAR（主指標） → Macro F1 → Accuracy（参考）。青：{train_curve_label}、橙：validation全件。</p>'
            f'<p>共通のbest epoch：{html.escape(str(training.get("best_epoch", "未記録")))}。選択基準：validation UAR → macro F1 → loss（完全同点は先のepoch）。</p>'
            + _figure_html(scores, "UAR（主指標）・Macro F1・Accuracy（参考）のscore曲線")
            + '<details><summary>lossを確認：split間の比較用／最適化に使用したloss</summary>'
            f'<p>比較用loss：{train_curve_label}とvalidation全件をそれぞれ重みなしで発話平均。−mean(log(clip(p_true, 1e−12, 1)))。</p>'
            f'<p>最適化loss：class_weighting={html.escape(str(weighting))}。各バッチのCrossEntropyLossを末尾バッチも含めて単純平均。重みありでは各バッチの対象ラベルの重み総和で正規化します。</p>'
            + _figure_html(losses, "比較用lossと最適化lossの2図")
            + f'<table><thead><tr><th>epoch</th><th>比較用{train_curve_label} loss</th><th>比較用validation loss</th><th>最適化train loss</th><th>best</th></tr></thead><tbody>'
            + ''.join(table) + '</tbody></table></details>'
            '<p>未記録の項目は欠測です。補間・ゼロ埋め・別指標の転用は行いません。</p>'
            '<p>train監視値が改善しvalidationが停滞・悪化する場合は過学習を疑う材料、両方のscoreが低いままなら学習不足などを調べる材料になります。監視値はcheckpoint選択や正式結果には使いません。</p>'
            + _best_training_metrics_html(training)
            + _best_validation_class_metrics_html(training)
            + _timing_html(training) + '</section>'
        )
    if display_output:
        display(result)
    return result


def load_saved_summary(path: str | Path) -> dict[str, Any]:
    """Read a completed JSON summary without cache validation or model operations."""
    source = Path(path)
    if not source.is_file():
        print("保存済みsummaryがありません:", source)
        return {"status": "summary_not_found", "summary_path": str(source)}
    return json.loads(source.read_text(encoding="utf-8"))


def mapping_summary() -> list[dict[str, Any]]:
    config = load_mapping_config()
    rows = []
    for dataset, contract in config["datasets"].items():
        for source, mapped in contract["mappings"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "mapping_version": contract["mapping_version"],
                    "original_emotion": source,
                    "mapped_emotion": mapped,
                    "included": True,
                    "approximate": source in contract.get("approximate_labels", []),
                }
            )
        for source in contract["excluded_labels"]:
            rows.append(
                {
                    "dataset": dataset,
                    "mapping_version": contract["mapping_version"],
                    "original_emotion": source,
                    "mapped_emotion": None,
                    "included": False,
                    "approximate": False,
                }
            )
    return rows


def split_summary() -> dict[str, Any]:
    return {
        "msp_podcast": {
            "version": MSP_SPLIT_VERSION,
            "assignment": {"Train": "train", "Development": "validation", "Test1": "test", "Test2": "excluded"},
        },
        "hcudb1": load_hcudb_split(),
        "iemocap": {"version": IEMOCAP_SPLIT_VERSION, "assignment": "all_sessions -> test"},
    }


def extraction_command_preview(
    manifest: str = "<manifest.jsonl>",
    audio_root: str = "<dataset-root>",
    cache_root: str = "<cache-root>",
) -> str:
    return (
        "python -m ser_pipeline extract-features "
        f"--manifest {manifest} --audio-root {audio_root} --cache-root {cache_root} "
        "--user-dir <upstream-dir> --checkpoint <base-checkpoint> --layer final --device auto"
    )


def one_item_feature_benchmark(feature_dim: int = 768, seconds: float = 1.0) -> dict[str, Any]:
    start = time.perf_counter()
    frames = max(1, int(seconds * 50))
    features = np.zeros((frames, feature_dim), dtype=np.float32)
    elapsed = time.perf_counter() - start
    return {
        "mode": "synthetic_preflight_only",
        "duration_seconds": float(seconds),
        "feature_frames": frames,
        "feature_dim": feature_dim,
        "feature_bytes": int(features.nbytes),
        "elapsed_seconds": float(elapsed),
    }


def _record(dataset: str, split: str, class_index: int, serial: int) -> dict[str, Any]:
    label = LABEL_ORDER[class_index]
    utterance_id = f"{dataset}_{split}_{class_index}_{serial}"
    audio_hash = hashlib.sha256(utterance_id.encode("utf-8")).hexdigest()
    mapping = {
        "msp_podcast": ("R1.10", "msp_podcast_r1_10_primary_v1", MSP_SPLIT_VERSION),
        "hcudb1": ("HCUDB1", "hcudb1_acted_emotion_v1", "hcudb1_speaker_split_v1"),
        "iemocap": ("IEMOCAP_full_release", "iemocap_external_v1", IEMOCAP_SPLIT_VERSION),
    }[dataset]
    originals = {
        "msp_podcast": ("A", "H", "S", "D"),
        "hcudb1": ("怒り", "狂喜・楽しい", "憂鬱・悲しい", "嫌い"),
        "iemocap": ("ang", "hap", "sad", "dis"),
    }[dataset]
    if dataset == "msp_podcast":
        source_split = {"train": "Train", "validation": "Development", "test": "Test1"}[split]
        speaker = f"msp_{split}_{serial}"
        session = "podcast"
    elif dataset == "hcudb1":
        source_split = "all"
        speaker = {"train": "FA", "validation": "FF", "test": "FG"}[split]
        session = ""
    else:
        source_split = "all_sessions"
        speaker = f"Ses0{serial % 5 + 1}F"
        session = f"Ses0{serial % 5 + 1}"
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_release": mapping[0],
        "utterance_id": utterance_id,
        "audio_relpath": f"unavailable/{utterance_id}.wav",
        "audio_sha256": audio_hash,
        "speaker_id": speaker,
        "speaker_id_status": "known",
        "group_id": speaker,
        "session_id": session,
        "source_split": source_split,
        "split": split,
        "split_version": mapping[2],
        "original_emotion": originals[class_index],
        "mapped_emotion": label,
        "class_index": class_index,
        "mapping_version": mapping[1],
        "included": True,
        "exclusion_reasons": [],
        "approximate_mapping": dataset == "hcudb1" and class_index == 3,
        "audio_size_bytes": 100,
        "sample_rate_hz": 16000,
        "channels": 1,
        "num_samples": 800,
        "duration_seconds": 0.05,
    }


def _demo_rows(dataset: str) -> list[dict[str, Any]]:
    if dataset == "iemocap":
        return [_record(dataset, "test", class_index, serial) for serial in range(2) for class_index in range(4)]
    rows = []
    for split, repeats in (("train", 2), ("validation", 1), ("test", 1)):
        for serial in range(repeats):
            for class_index in range(4):
                rows.append(_record(dataset, split, class_index, serial))
    return rows


def _demo_features(row: dict[str, Any], feature_dim: int) -> np.ndarray:
    frames = 3 + int(hashlib.sha256(row["utterance_id"].encode()).digest()[0] % 3)
    vector = np.zeros(feature_dim, dtype=np.float32)
    vector[int(row["class_index"])] = 3.0
    vector[4:] = (int(row["class_index"]) + 1) / 10.0
    return np.repeat(vector[None, :], frames, axis=0)


def _write_demo_cache(cache_root: Path, manifest_path: Path, rows: list[dict[str, Any]], feature_dim: int) -> None:
    encoder_hash = "d" * 64
    for split in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == split]
        directory = cache_root / rows[0]["dataset"] / split
        directory.mkdir(parents=True, exist_ok=True)
        arrays = [_demo_features(row, feature_dim) for row in subset]
        concatenated = np.concatenate(arrays, axis=0).astype(np.float32)
        shard_name = "shard-00000.npy"
        index_name = "shard-00000.index.jsonl"
        shard_path = directory / shard_name
        np.save(shard_path, concatenated, allow_pickle=False)
        entries = []
        offset = 0
        for row, array in zip(subset, arrays):
            entries.append(
                CacheIndexEntry(
                    dataset=row["dataset"],
                    split=split,
                    utterance_id=row["utterance_id"],
                    shard=shard_name,
                    offset=offset,
                    num_frames=len(array),
                    feature_dim=feature_dim,
                    class_index=row["class_index"],
                )
            )
            offset += len(array)
        index_path = directory / index_name
        _write_index(entries, index_path)
        shard_meta = {
            "shard": shard_name,
            "index": index_name,
            "shard_sha256": sha256_file(shard_path),
            "index_sha256": sha256_file(index_path),
            "frames": int(concatenated.shape[0]),
            "utterances": len(entries),
            "feature_dim": feature_dim,
            "dtype": "float32",
        }
        _atomic_json(shard_meta, directory / "shard-00000.meta.json")
        _atomic_json(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "dataset": rows[0]["dataset"],
                "split": split,
                "utterance_count": len(entries),
                "shards": [shard_meta],
            },
            directory / "_SUCCESS",
        )
    mapping_versions = sorted({row["mapping_version"] for row in rows})
    split_versions = sorted({row["split_version"] for row in rows})
    cache_id = hashlib.sha256((rows[0]["dataset"] + str(feature_dim)).encode()).hexdigest()[:16]
    _atomic_json(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_id": cache_id,
            "encoder_name": "synthetic_demo_encoder",
            "encoder_checkpoint_sha256": encoder_hash,
            "feature_layer": FEATURE_LAYER,
            "feature_dim": feature_dim,
            "dtype": "float32",
            "extraction_code_version": EXTRACTION_CODE_VERSION,
            "git_commit": "demo",
            "manifest_sha256": manifest_sha256(manifest_path),
            "exclusion_contract": None,
            "duplicate_audit": None,
            "duplicate_exclusion_contract": None,
            "mapping_versions": mapping_versions,
            "split_versions": split_versions,
            "audio_preprocessing": {
                "target_sample_rate_hz": 16000,
                "channels": "mono_required",
                "resampler": "scipy.signal.resample_poly",
            },
            "shard_policy": {"max_frames_approximately": 65536},
            "complete": True,
        },
        cache_root / "cache_meta.json",
    )


def make_demo_artifacts(
    root: str | Path,
    feature_dim: int = 8,
    *,
    datasets: Sequence[str] = ("msp_podcast", "hcudb1", "iemocap"),
) -> dict[str, DatasetArtifacts]:
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for dataset in datasets:
        rows = _demo_rows(dataset)
        manifest_path = destination / dataset / "manifest.jsonl"
        write_manifest(rows, manifest_path)
        cache_root = destination / dataset / "cache"
        _write_demo_cache(cache_root, manifest_path, rows, feature_dim)
        validate_cache(cache_root, manifest_path)
        artifacts[dataset] = DatasetArtifacts(manifest_path=manifest_path, cache_root=cache_root)
    return artifacts


def demo_cache_summary(
    root: str | Path,
    *,
    datasets: Sequence[str] = ("msp_podcast", "hcudb1", "iemocap"),
) -> dict[str, Any]:
    artifacts = make_demo_artifacts(root, datasets=datasets)
    return {
        dataset: validate_cache(current.cache_root, current.manifest_path)
        for dataset, current in artifacts.items()
    }


def run_demo_transfer_study(
    root: str | Path,
    *,
    seeds=STUDY_SEEDS,
    epochs: int = 1,
) -> dict[str, Any]:
    destination = Path(root)
    artifacts = make_demo_artifacts(destination / "artifacts", datasets=EVALUATION_DATASETS)
    config = TrainingConfig(
        seed=int(seeds[0]),
        device="cpu",
        epochs=epochs,
        batch_size=4,
        learning_rate=0.01,
        hidden_dim=8,
        dropout=0.0,
    )
    return run_transfer_study(artifacts, destination / "study", seeds=seeds, base_config=config)


__all__ = [
    "STUDY_SEEDS",
    "demo_cache_summary",
    "environment_summary",
    "extraction_command_preview",
    "make_demo_artifacts",
    "mapping_summary",
    "one_item_feature_benchmark",
    "plot_training_scores",
    "plot_training_losses",
    "display_training_history",
    "load_saved_summary",
    "run_demo_transfer_study",
    "split_summary",
]
