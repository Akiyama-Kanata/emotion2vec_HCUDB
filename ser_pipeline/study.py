"""MSP parent -> before evaluation -> HCUDB child -> after evaluation study."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audio import sha256_file
from .checkpoints import load_decoder_checkpoint
from .evaluation import assert_same_evaluation_sets
from .training import TrainingConfig, evaluate_checkpoint, train_decoder


STUDY_SEEDS = (42, 43, 44)
EVALUATION_DATASETS = ("msp_podcast", "hcudb1", "iemocap")


@dataclass(frozen=True)
class DatasetArtifacts:
    manifest_path: Path
    cache_root: Path


def _artifact(artifacts: Mapping[str, DatasetArtifacts], dataset: str) -> DatasetArtifacts:
    try:
        return artifacts[dataset]
    except KeyError as exc:
        raise ValueError(f"study artifacts are missing dataset: {dataset}") from exc


def run_transfer_study(
    artifacts: Mapping[str, DatasetArtifacts],
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = STUDY_SEEDS,
    base_config: TrainingConfig | None = None,
) -> dict[str, Any]:
    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("study seeds must be a non-empty unique sequence")
    for dataset in EVALUATION_DATASETS:
        _artifact(artifacts, dataset)
    template = base_config or TrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        config = replace(template, seed=seed)
        seed_dir = output / f"seed-{seed}"
        msp = _artifact(artifacts, "msp_podcast")
        parent = train_decoder(
            msp.manifest_path,
            msp.cache_root,
            "msp_podcast",
            seed_dir / "checkpoints" / "msp",
            config,
            training_stage="msp_train",
        )
        parent_path = Path(parent["best_checkpoint"])

        before: dict[str, Any] = {}
        for dataset in EVALUATION_DATASETS:
            current = _artifact(artifacts, dataset)
            before[dataset] = evaluate_checkpoint(
                parent_path,
                current.manifest_path,
                current.cache_root,
                dataset,
                seed_dir / "before" / dataset,
                batch_size=config.batch_size,
                device=config.device,
            )

        hcudb = _artifact(artifacts, "hcudb1")
        child = train_decoder(
            hcudb.manifest_path,
            hcudb.cache_root,
            "hcudb1",
            seed_dir / "checkpoints" / "hcudb",
            config,
            training_stage="hcudb_continue",
            parent_checkpoint=parent_path,
        )
        child_path = Path(child["best_checkpoint"])
        child_payload = load_decoder_checkpoint(child_path)
        parent_payload = load_decoder_checkpoint(parent_path)
        if child_payload["parent_checkpoint_id"] != parent_payload["checkpoint_id"]:
            raise ValueError("child checkpoint parent ID mismatch")
        if child_payload["parent_checkpoint_sha256"] != sha256_file(parent_path):
            raise ValueError("child checkpoint parent SHA-256 mismatch")

        after: dict[str, Any] = {}
        for dataset in EVALUATION_DATASETS:
            current = _artifact(artifacts, dataset)
            after[dataset] = evaluate_checkpoint(
                child_path,
                current.manifest_path,
                current.cache_root,
                dataset,
                seed_dir / "after" / dataset,
                batch_size=config.batch_size,
                device=config.device,
            )
            assert_same_evaluation_sets(
                before[dataset]["result"]["set_signature"],
                after[dataset]["result"]["set_signature"],
            )
        runs.append(
            {
                "seed": seed,
                "parent": parent,
                "child": child,
                "before": before,
                "after": after,
            }
        )
    summary = {
        "seeds": [int(seed) for seed in seeds],
        "evaluation_datasets": list(EVALUATION_DATASETS),
        "runs": runs,
    }
    summary_path = output / "study_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


run_msp_hcudb_study = run_transfer_study


__all__ = [
    "DatasetArtifacts",
    "EVALUATION_DATASETS",
    "STUDY_SEEDS",
    "run_msp_hcudb_study",
    "run_transfer_study",
]
