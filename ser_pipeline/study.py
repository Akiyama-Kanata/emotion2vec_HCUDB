"""MSP parent -> before evaluation -> HCUDB child -> after evaluation study."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audio import sha256_file
from .checkpoints import load_decoder_checkpoint
from .evaluation import assert_same_evaluation_sets
from .exclusions import (
    load_msp_missing_audio_exclusion_contract,
    manifest_exclusion_contract_signature,
    write_msp_missing_audio_exclusion_contract,
)
from .manifest import load_manifest, manifest_sha256
from .training import TrainingConfig, evaluate_checkpoint, train_decoder


STUDY_SEEDS = (42, 43, 44)
EVALUATION_DATASETS = ("msp_podcast", "hcudb1")


@dataclass(frozen=True)
class DatasetArtifacts:
    manifest_path: Path
    cache_root: Path
    exclusion_contract_path: Path | None = None


def require_formal_epochs(epochs: int | None) -> int:
    """Return an explicitly configured positive epoch count for a formal run."""
    if epochs is None:
        raise ValueError("formal epochs must be set explicitly before execution")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("formal epochs must be a positive integer")
    return epochs


def _artifact(artifacts: Mapping[str, DatasetArtifacts], dataset: str) -> DatasetArtifacts:
    try:
        return artifacts[dataset]
    except KeyError as exc:
        raise ValueError(f"study artifacts are missing dataset: {dataset}") from exc


def bundle_msp_exclusion_contract(
    artifact: DatasetArtifacts,
    output_dir: str | Path,
) -> dict[str, Any] | None:
    """Copy a validated MSP contract into study provenance and link its downstream IDs."""
    signature = manifest_exclusion_contract_signature(load_manifest(artifact.manifest_path))
    if signature is None:
        if artifact.exclusion_contract_path is not None:
            raise ValueError("an MSP exclusion contract was supplied for a manifest without contract provenance")
        return None
    if artifact.exclusion_contract_path is None:
        raise ValueError("MSP manifest contract provenance requires exclusion_contract_path")
    payload, report = load_msp_missing_audio_exclusion_contract(
        artifact.exclusion_contract_path,
        expected_sha256=signature["normalized_sha256"],
    )
    cache_meta_path = artifact.cache_root / "cache_meta.json"
    try:
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MSP cache metadata: {cache_meta_path}") from exc
    if cache_meta.get("exclusion_contract") != signature:
        raise ValueError("MSP cache and manifest exclusion contract provenance differ")
    destination = Path(output_dir) / "provenance" / "msp_missing_audio_exclusions_v1.json"
    write_msp_missing_audio_exclusion_contract(payload, destination)
    return {
        "path": str(destination),
        "normalized_sha256": report["normalized_sha256"],
        "manifest_sha256": manifest_sha256(artifact.manifest_path),
        "cache_id": cache_meta.get("cache_id"),
        "final_included": signature["final_included"],
    }


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
    exclusion_contract_artifact = bundle_msp_exclusion_contract(
        _artifact(artifacts, "msp_podcast"),
        output,
    )
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
        parent_sha256 = sha256_file(parent_path)
        child_sha256 = sha256_file(child_path)
        if child_payload["parent_checkpoint_id"] != parent_payload["checkpoint_id"]:
            raise ValueError("child checkpoint parent ID mismatch")
        if child_payload["parent_checkpoint_sha256"] != parent_sha256:
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
                "provenance": {
                    "parent_checkpoint": {
                        "id": parent_payload["checkpoint_id"],
                        "sha256": parent_sha256,
                        "cache_id": parent_payload["cache_id"],
                        "path": str(parent_path),
                    },
                    "child_checkpoint": {
                        "id": child_payload["checkpoint_id"],
                        "sha256": child_sha256,
                        "cache_id": child_payload["cache_id"],
                        "path": str(child_path),
                        "parent_id": child_payload["parent_checkpoint_id"],
                        "parent_sha256": child_payload["parent_checkpoint_sha256"],
                    },
                    "evaluation_sets": {
                        dataset: before[dataset]["result"]["set_signature"]
                        for dataset in EVALUATION_DATASETS
                    },
                    "exclusion_contract_artifact": exclusion_contract_artifact,
                    "training_configs": {
                        "parent": parent["config"],
                        "child": child["config"],
                    },
                },
            }
        )
    summary = {
        "seeds": [int(seed) for seed in seeds],
        "evaluation_datasets": list(EVALUATION_DATASETS),
        "exclusion_contract_artifact": exclusion_contract_artifact,
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
    "bundle_msp_exclusion_contract",
    "require_formal_epochs",
    "run_msp_hcudb_study",
    "run_transfer_study",
]
