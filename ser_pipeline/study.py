"""MSP parent -> before evaluation -> HCUDB child -> after evaluation study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .audio import sha256_file
from .cache import ShardedFeatureStore, _atomic_json
from .checkpoints import decoder_signature, load_decoder_checkpoint
from .evaluation import assert_same_evaluation_sets, evaluation_set_signature
from .duplicates import (
    load_msp_audio_duplicate_audit,
    load_msp_audio_duplicate_exclusion_contract,
    manifest_duplicate_provenance_signature,
    write_msp_audio_duplicate_audit,
    write_msp_audio_duplicate_exclusion_contract,
)
from .exclusions import (
    load_msp_missing_audio_exclusion_contract,
    manifest_exclusion_contract_signature,
    write_msp_missing_audio_exclusion_contract,
)
from .manifest import load_manifest, manifest_sha256
from .model import BaseModel
from .training import TrainingConfig, evaluate_checkpoint, train_decoder, training_loss_config


STUDY_SEEDS = (42, 43, 44)
EVALUATION_DATASETS = ("msp_podcast", "hcudb1")


@dataclass(frozen=True)
class DatasetArtifacts:
    manifest_path: Path
    cache_root: Path
    exclusion_contract_path: Path | None = None
    duplicate_audit_path: Path | None = None
    duplicate_exclusion_contract_path: Path | None = None


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


def prepare_study_stores(
    artifacts: Mapping[str, DatasetArtifacts],
    stores: Mapping[str, ShardedFeatureStore] | None = None,
) -> dict[str, ShardedFeatureStore]:
    """Validate once for this study invocation, including the notebook's entry gate."""
    prepared = {}
    for dataset in EVALUATION_DATASETS:
        current = _artifact(artifacts, dataset)
        if stores is None:
            print(f"[cache {dataset}] full validation started", flush=True)
            store = ShardedFeatureStore(current.cache_root, current.manifest_path)
            print(f"[cache {dataset}] validated in {store.validation_seconds:.2f}s", flush=True)
        else:
            if dataset not in stores:
                raise ValueError(f"study feature stores are missing dataset: {dataset}")
            store = stores[dataset]
            store.require_paths(current.cache_root, current.manifest_path)
            store.ensure_validated()
        prepared[dataset] = store
    return prepared


def summarize_study(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Small notebook display; complete results remain in the returned object/files."""
    if "runs" not in summary:
        return dict(summary)
    rows = []
    for run in summary["runs"]:
        row: dict[str, Any] = {"seed": run["seed"]}
        for stage in ("parent", "child"):
            training = run[stage]
            row[stage] = {
                "best_epoch": training["best_epoch"],
                "validation": {
                    key: training["best_validation_metrics"][key]
                    for key in ("uar", "macro_f1", "wa", "loss")
                },
                "seconds": training.get("timings", {}).get("total_seconds"),
                "checkpoint": training["best_checkpoint"],
            }
        for stage in ("before", "after"):
            row[stage] = {
                dataset: {
                    **{key: evaluation["result"]["metrics_4class"][key] for key in ("uar", "macro_f1", "wa", "loss")},
                    "seconds": evaluation.get("timings", {}).get("total_seconds"),
                    "paths": evaluation["paths"],
                }
                for dataset, evaluation in run[stage].items()
            }
        rows.append(row)
    return {
        "seeds": summary["seeds"],
        "runs": rows,
        "timings": summary.get("timings"),
        "timings_path": summary.get("timings_path"),
        "summary_path": summary.get("summary_path"),
    }


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


def bundle_msp_duplicate_provenance(
    artifact: DatasetArtifacts,
    output_dir: str | Path,
) -> dict[str, Any] | None:
    """Copy validated duplicate audit and exclusion contracts into study provenance."""
    signature = manifest_duplicate_provenance_signature(load_manifest(artifact.manifest_path))
    supplied = (artifact.duplicate_audit_path, artifact.duplicate_exclusion_contract_path)
    if signature is None:
        if any(path is not None for path in supplied):
            raise ValueError("duplicate artifacts were supplied for a manifest without duplicate provenance")
        return None
    if any(path is None for path in supplied):
        raise ValueError("MSP manifest duplicate provenance requires both audit and exclusion contract paths")
    audit_payload, audit_report = load_msp_audio_duplicate_audit(
        artifact.duplicate_audit_path,
        expected_sha256=signature["audit"]["normalized_sha256"],
    )
    contract_payload, contract_report = load_msp_audio_duplicate_exclusion_contract(
        artifact.duplicate_exclusion_contract_path,
        audit_payload,
        expected_sha256=signature["exclusion_contract"]["normalized_sha256"],
    )
    cache_meta_path = artifact.cache_root / "cache_meta.json"
    try:
        cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MSP cache metadata: {cache_meta_path}") from exc
    if cache_meta.get("duplicate_audit") != signature["audit"]:
        raise ValueError("MSP cache and manifest duplicate audit provenance differ")
    if cache_meta.get("duplicate_exclusion_contract") != signature["exclusion_contract"]:
        raise ValueError("MSP cache and manifest duplicate exclusion provenance differ")
    provenance_dir = Path(output_dir) / "provenance"
    audit_destination = provenance_dir / "msp_audio_duplicate_audit_v1.json"
    contract_destination = provenance_dir / "msp_audio_duplicate_exclusions_v1.json"
    write_msp_audio_duplicate_audit(audit_payload, audit_destination)
    write_msp_audio_duplicate_exclusion_contract(contract_payload, audit_payload, contract_destination)
    return {
        "audit": {
            "path": str(audit_destination),
            "normalized_sha256": audit_report["normalized_sha256"],
        },
        "exclusion_contract": {
            "path": str(contract_destination),
            "normalized_sha256": contract_report["normalized_sha256"],
            "count": contract_report["count"],
            "final_included": contract_report["post_exclusion_counts"]["final_included"],
        },
        "manifest_sha256": manifest_sha256(artifact.manifest_path),
        "cache_id": cache_meta.get("cache_id"),
    }


def run_transfer_study(
    artifacts: Mapping[str, DatasetArtifacts],
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = STUDY_SEEDS,
    base_config: TrainingConfig | None = None,
    stores: Mapping[str, ShardedFeatureStore] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("study seeds must be a non-empty unique sequence")
    for dataset in EVALUATION_DATASETS:
        _artifact(artifacts, dataset)
    template = base_config or TrainingConfig()
    # Scoped to this invocation. No global registry or on-disk validation token.
    stores = prepare_study_stores(artifacts, stores)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    exclusion_contract_artifact = bundle_msp_exclusion_contract(
        _artifact(artifacts, "msp_podcast"),
        output,
    )
    duplicate_provenance_artifact = bundle_msp_duplicate_provenance(
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
            store=stores["msp_podcast"],
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
                store=stores[dataset],
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
            store=stores["hcudb1"],
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
                store=stores[dataset],
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
                    "duplicate_provenance_artifact": duplicate_provenance_artifact,
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
        "duplicate_provenance_artifact": duplicate_provenance_artifact,
        "runs": runs,
        "cache_validation": {name: store.validation_report for name, store in stores.items()},
        "timings": {
            "cache_validation": {
                name: {"seconds": store.validation_seconds, "full_passes": store.validation_count}
                for name, store in stores.items()
            },
            "before_evaluation_seconds": sum(
                evaluation["timings"]["evaluation_seconds"]
                for run in runs for evaluation in run["before"].values()
            ),
            "after_evaluation_seconds": sum(
                evaluation["timings"]["evaluation_seconds"]
                for run in runs for evaluation in run["after"].values()
            ),
        },
    }
    summary_path = output / "study_summary.json"
    timing_path = output / "study_timings.json"
    summary["summary_path"] = str(summary_path)
    summary["timings_path"] = str(timing_path)
    save_started = perf_counter()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["timings"]["summary_save_seconds"] = perf_counter() - save_started
    summary["timings"]["study_seconds"] = perf_counter() - started
    _atomic_json(summary["timings"], timing_path)
    print(f"[study] total={summary['timings']['study_seconds']:.2f}s output={summary_path}", flush=True)
    return summary


run_msp_hcudb_study = run_transfer_study


def load_msp_comparison_baselines(
    summary_paths: Sequence[str | Path],
    store: ShardedFeatureStore,
    config: TrainingConfig,
    seeds: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Verify saved unweighted MSP runs before a validation-only loss comparison."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("comparison seeds must be non-empty and unique")
    if config.class_weighting != "none" or config.patience is not None:
        raise ValueError("comparison base config requires unweighted loss and no early stopping")
    store.ensure_validated()
    manifest_hash = manifest_sha256(store.manifest_path)
    baselines = {}
    for summary_path in summary_paths:
        path = Path(summary_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        for run in summary.get("runs", []):
            seed = int(run["seed"])
            if seed not in seeds:
                continue
            if seed in baselines:
                raise ValueError(f"multiple baselines supplied for seed {seed}")
            parent = run["parent"]
            expected_config = replace(config, seed=seed)
            recorded_config = dict(parent["config"])
            recorded_config.setdefault("class_weighting", "none")
            if recorded_config != asdict(expected_config):
                raise ValueError(f"baseline training configuration mismatch for seed {seed}")
            if parent["training_stage"] != "msp_train" or parent["dataset"] != "msp_podcast":
                raise ValueError("baseline must be an MSP parent training run")
            if [row["epoch"] for row in parent["history"]] != list(range(1, config.epochs + 1)):
                raise ValueError(f"baseline does not contain all configured epochs for seed {seed}")
            provenance = run["provenance"]
            if provenance["evaluation_sets"]["msp_podcast"]["manifest_sha256"] != manifest_hash:
                raise ValueError("baseline manifest mismatch (training/validation sets must be unchanged)")
            checkpoint_info = provenance["parent_checkpoint"]
            if checkpoint_info["cache_id"] != store.meta["cache_id"]:
                raise ValueError("baseline feature cache ID mismatch")
            checkpoint_path = Path(parent["best_checkpoint"])
            if not checkpoint_path.is_file():
                # Allow moving a complete study directory between Windows and WSL.
                checkpoint_path = path.parent / f"seed-{seed}" / "checkpoints" / "msp" / f"msp_train_seed{seed}_best.pt"
            if sha256_file(checkpoint_path) != checkpoint_info["sha256"]:
                raise ValueError("baseline checkpoint SHA-256 mismatch")
            model = BaseModel(
                input_dim=int(store.meta["feature_dim"]), hidden_dim=config.hidden_dim, dropout=config.dropout,
            )
            payload = load_decoder_checkpoint(
                checkpoint_path, expected_signature=decoder_signature(model, seed, store.meta),
                expected_stage="msp_train",
            )
            if payload["cache_id"] != store.meta["cache_id"] or payload["checkpoint_id"] != checkpoint_info["id"]:
                raise ValueError("baseline checkpoint provenance mismatch")
            saved_loss = payload.get("loss_config")
            if saved_loss is not None and saved_loss["class_weighting"] != "none":
                raise ValueError("baseline checkpoint must use unweighted loss")
            if payload["epoch"] != parent["best_epoch"] or payload["validation_metrics"] != parent["best_validation_metrics"]:
                raise ValueError("baseline summary and best checkpoint validation results differ")
            baselines[seed] = {"training": parent, "summary_path": str(path), "checkpoint_path": str(checkpoint_path)}
    if set(baselines) != set(seeds):
        raise ValueError(f"missing baseline seeds: {sorted(set(seeds) - set(baselines))}")
    return baselines


def run_msp_loss_comparison(
    artifact: DatasetArtifacts,
    output_dir: str | Path,
    baseline_summary_paths: Sequence[str | Path],
    *,
    seeds: Sequence[int] = (42,),
    base_config: TrainingConfig | None = None,
    store: ShardedFeatureStore | None = None,
) -> dict[str, Any]:
    """Train weighted MSP models from scratch and compare saved validation results only."""
    started = perf_counter()
    template = base_config or TrainingConfig(epochs=10, device="cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"comparison output is not empty; choose a new output directory: {output}")
    if store is None:
        print("[MSP comparison] full cache validation started", flush=True)
        store = ShardedFeatureStore(artifact.cache_root, artifact.manifest_path)
    else:
        store.require_paths(artifact.cache_root, artifact.manifest_path)
    baselines = load_msp_comparison_baselines(baseline_summary_paths, store, template, seeds)
    loss_config = training_loss_config(store, "msp_podcast", "balanced")
    validation_signature = evaluation_set_signature(artifact.manifest_path, "msp_podcast", "validation")
    exclusion = bundle_msp_exclusion_contract(artifact, output)
    duplicates = bundle_msp_duplicate_provenance(artifact, output)
    rows = []
    runs = []
    summary_path = output / "comparison_summary.json"
    for seed in seeds:
        baseline = baselines[seed]["training"]
        weighted = train_decoder(
            artifact.manifest_path, artifact.cache_root, "msp_podcast", output / f"seed-{seed}" / "balanced",
            replace(template, seed=seed, class_weighting="balanced"), training_stage="msp_train", store=store,
        )
        metrics_before = baseline["best_validation_metrics"]
        metrics_after = weighted["best_validation_metrics"]
        for condition, result in (("none", baseline), ("balanced", weighted)):
            metrics = result["best_validation_metrics"]
            row = {"seed": seed, "condition": condition, "best_epoch": result["best_epoch"]}
            row.update({key: metrics[key] for key in ("uar", "macro_f1", "wa", "loss")})
            row.update({f"recall_{item['class_label']}": item["recall"] for item in metrics["class_metrics"]})
            rows.append(row)
        deltas = {key: metrics_after[key] - metrics_before[key] for key in ("uar", "macro_f1", "wa", "loss")}
        runs.append({"seed": seed, "baseline": baselines[seed], "weighted": weighted, "validation_deltas": deltas})
        summary = {
            "dataset": "msp_podcast", "selection_split": "validation", "test_evaluated": False,
            "requested_seeds": list(seeds), "completed_seeds": [run["seed"] for run in runs],
            "base_config": asdict(template), "loss_config": loss_config,
            "cache_id": store.meta["cache_id"], "validation_signature": validation_signature,
            "exclusion_contract_artifact": exclusion, "duplicate_provenance_artifact": duplicates,
            "rows": rows, "runs": runs, "summary_path": str(summary_path),
            "seconds": perf_counter() - started,
        }
        _atomic_json(summary, summary_path)
        print(f"[MSP comparison seed={seed}] validation changes: {deltas}", flush=True)
    return summary


__all__ = [
    "DatasetArtifacts",
    "EVALUATION_DATASETS",
    "STUDY_SEEDS",
    "bundle_msp_exclusion_contract",
    "bundle_msp_duplicate_provenance",
    "require_formal_epochs",
    "prepare_study_stores",
    "summarize_study",
    "run_msp_hcudb_study",
    "run_transfer_study",
    "load_msp_comparison_baselines",
    "run_msp_loss_comparison",
]
