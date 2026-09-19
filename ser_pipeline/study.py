"""MSP/HCUDB training studies and explicitly selected, separate final test evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .audio import sha256_file
from .cache import ShardedFeatureStore, _atomic_json
from .checkpoints import decoder_signature, load_decoder_checkpoint, validate_signature
from .evaluation import assert_same_evaluation_sets, evaluation_set_signature
from .duplicates import (
    load_msp_audio_duplicate_audit,
    load_msp_audio_duplicate_exclusion_contract,
    manifest_duplicate_provenance_signature,
    write_msp_audio_duplicate_audit,
    write_msp_audio_duplicate_exclusion_contract,
)
from .diagnostics import (
    OfficialTrainingDiagnosticsConfig,
    aggregate_official_training_diagnostics,
)
from .exclusions import (
    load_msp_missing_audio_exclusion_contract,
    manifest_exclusion_contract_signature,
    write_msp_missing_audio_exclusion_contract,
)
from .manifest import load_manifest, manifest_sha256
from .model import BaseModel
from .training import (
    TrainingConfig,
    TrainingMonitoringConfig,
    evaluate_checkpoint,
    resolve_device,
    selection_key,
    train_decoder,
    training_loss_config,
)


STUDY_SEEDS = (42, 43, 44)
EVALUATION_DATASETS = ("msp_podcast", "hcudb1")


@dataclass(frozen=True)
class DatasetArtifacts:
    manifest_path: Path
    cache_root: Path
    exclusion_contract_path: Path | None = None
    duplicate_audit_path: Path | None = None
    duplicate_exclusion_contract_path: Path | None = None


@dataclass(frozen=True)
class FinalEvaluationTarget:
    """A user-selected best checkpoint, pinned by SHA-256, and its test dataset."""

    name: str
    checkpoint_path: Path
    expected_sha256: str
    dataset: str


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
                "train": {
                    key: training["best_training_metrics"].get(key)
                    for key in ("uar", "macro_f1", "wa")
                } if training.get("best_training_metrics") is not None else None,
                "validation": {
                    key: training["best_validation_metrics"][key]
                    for key in ("uar", "macro_f1", "wa")
                },
                "seconds": training.get("timings", {}).get("total_seconds"),
                "checkpoint": training["best_checkpoint"],
            }
        rows.append(row)
    return {
        "seeds": summary["seeds"],
        "test_evaluated": summary.get("test_evaluated", any("before" in run or "after" in run for run in summary["runs"])),
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
    monitoring_config: TrainingMonitoringConfig | None = None,
    stores: Mapping[str, ShardedFeatureStore] | None = None,
) -> dict[str, Any]:
    """Train MSP parents and HCUDB children using train/validation only; never run test."""
    started = perf_counter()
    if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("study seeds must be a non-empty unique sequence")
    for dataset in EVALUATION_DATASETS:
        _artifact(artifacts, dataset)
    template = base_config or TrainingConfig()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"study output is not empty; choose a new output directory: {output}")
    # Scoped to this invocation. No global registry or on-disk validation token.
    stores = prepare_study_stores(artifacts, stores)
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
    training_sets = {
        dataset: {split: evaluation_set_signature(_artifact(artifacts, dataset).manifest_path, dataset, split)
                  for split in ("train", "validation")}
        for dataset in EVALUATION_DATASETS
    }
    monitoring_kwargs = {"monitoring_config": monitoring_config} if monitoring_config is not None else {}
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
            **monitoring_kwargs,
        )
        parent_path = Path(parent["best_checkpoint"])

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
            **monitoring_kwargs,
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

        runs.append(
            {
                "seed": seed,
                "parent": parent,
                "child": child,
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
                    "training_sets": training_sets,
                    "exclusion_contract_artifact": exclusion_contract_artifact,
                    "duplicate_provenance_artifact": duplicate_provenance_artifact,
                    "training_configs": {
                        "parent": parent["config"],
                        "child": child["config"],
                    },
                    "monitoring_config": asdict(monitoring_config) if monitoring_config is not None else None,
                },
            }
        )
    summary = {
        "seeds": [int(seed) for seed in seeds],
        "evaluation_datasets": list(EVALUATION_DATASETS),
        "test_evaluated": False,
        "selection_split": "validation",
        "monitoring_config": asdict(monitoring_config) if monitoring_config is not None else None,
        "training_sets": training_sets,
        "exclusion_contract_artifact": exclusion_contract_artifact,
        "duplicate_provenance_artifact": duplicate_provenance_artifact,
        "runs": runs,
        "cache_validation": {name: store.validation_report for name, store in stores.items()},
        "timings": {
            "cache_validation": {
                name: {"seconds": store.validation_seconds, "full_passes": store.validation_count}
                for name, store in stores.items()
            },
        },
    }
    summary_path = output / "study_summary.json"
    timing_path = output / "study_timings.json"
    summary["summary_path"] = str(summary_path)
    summary["timings_path"] = str(timing_path)
    save_started = perf_counter()
    summary["timings"]["study_seconds"] = perf_counter() - started
    _atomic_json(summary, summary_path)
    summary["timings"]["summary_save_seconds"] = perf_counter() - save_started
    summary["timings"]["study_seconds"] = perf_counter() - started
    _atomic_json(summary, summary_path)
    _atomic_json(summary["timings"], timing_path)
    print(f"[study] total={summary['timings']['study_seconds']:.2f}s output={summary_path}", flush=True)
    return summary


run_msp_hcudb_study = run_transfer_study


def run_final_evaluations(
    artifacts: Mapping[str, DatasetArtifacts],
    targets: Sequence[FinalEvaluationTarget],
    output_dir: str | Path,
    *,
    device: str,
    batch_size: int = 8,
    stores: Mapping[str, ShardedFeatureStore] | None = None,
) -> dict[str, Any]:
    """Validate all pinned best checkpoints, persist the plan, then evaluate test.

    This entry point never trains, substitutes checkpoints, or selects using test.
    An MSP-only target needs only MSP artifacts. A transfer study explicitly lists
    both parent and child on both datasets. Existing output directories are not reused.
    """
    if not targets:
        raise ValueError("final evaluation targets must be explicitly selected")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"final evaluation output is not empty: {output}")
    selected_device = resolve_device(device)
    selected = []
    identities = set()
    for target in targets:
        if not target.name.strip() or target.dataset not in EVALUATION_DATASETS:
            raise ValueError("final target requires a display name and supported dataset")
        current = _artifact(artifacts, target.dataset)
        checkpoint = Path(target.checkpoint_path).resolve(strict=True)
        expected_sha = target.expected_sha256.lower()
        if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            raise ValueError("final target requires an explicit SHA-256")
        if sha256_file(checkpoint) != expected_sha:
            raise ValueError("final checkpoint SHA-256 mismatch")
        identity = (checkpoint, target.dataset)
        if identity in identities:
            raise ValueError("duplicate final checkpoint/dataset target")
        identities.add(identity)
        payload = load_decoder_checkpoint(checkpoint)
        if payload.get("selection") != "best_validation":
            raise ValueError("final evaluation requires a best validation checkpoint")
        if payload.get("best_epoch") not in (None, payload["epoch"]):
            raise ValueError("final checkpoint epoch differs from best epoch")
        if payload.get("best_validation_metrics") is not None and payload["best_validation_metrics"] != payload["validation_metrics"]:
            raise ValueError("final checkpoint validation metrics differ from best metrics")
        scored = [entry for entry in payload["history"] if isinstance(entry, dict) and entry.get("validation")]
        if scored:
            best = max(sorted(scored, key=lambda entry: entry["epoch"]), key=lambda entry: selection_key(entry["validation"]))
            if best["epoch"] != payload["epoch"] or best["validation"] != payload["validation_metrics"]:
                raise ValueError("final checkpoint does not match recorded validation selection")
        selected.append((target, current, checkpoint, payload, expected_sha))
    prepared = {}
    plan_targets = []
    for index, (target, current, checkpoint, payload, expected_sha) in enumerate(selected, 1):
        if target.dataset not in prepared:
            if stores is None:
                prepared[target.dataset] = ShardedFeatureStore(current.cache_root, current.manifest_path)
            else:
                if target.dataset not in stores:
                    raise ValueError(f"final feature stores are missing dataset: {target.dataset}")
                store = stores[target.dataset]
                store.require_paths(current.cache_root, current.manifest_path)
                store.ensure_validated()
                prepared[target.dataset] = store
        store = prepared[target.dataset]
        model = BaseModel(**payload["signature"]["model_config"])
        expected = decoder_signature(model, int(payload["signature"]["seed"]), store.meta)
        validate_signature(payload["signature"], expected, context="final evaluation")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        plan_targets.append({
            "name": target.name, "checkpoint_path": str(checkpoint), "expected_sha256": expected_sha,
            "dataset": target.dataset, "split": "test", "best_epoch": payload["epoch"],
            "checkpoint_id": payload["checkpoint_id"], "training_stage": payload["training_stage"],
            "signature": payload["signature"], "loss_config": payload.get("loss_config"),
            "parent_checkpoint_id": payload.get("parent_checkpoint_id"),
            "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
            "cache_id": store.meta["cache_id"], "manifest_path": str(current.manifest_path),
            "set_signature": evaluation_set_signature(current.manifest_path, target.dataset, "test"),
            "output_dir": str(output / f"target-{index:03d}" / target.dataset),
        })
    plan = {"device": str(selected_device), "batch_size": batch_size, "targets": plan_targets, "selection_split": "validation"}
    plan_path = output / "final_evaluation_plan.json"
    _atomic_json(plan, plan_path)  # Written before the first test prediction.
    summary_path = output / "final_evaluation_summary.json"
    summary = {"plan_path": str(plan_path), "summary_path": str(summary_path), "status": "running", "test_evaluated": False, "evaluations": []}
    _atomic_json(summary, summary_path)
    try:
        for record in plan_targets:
            if sha256_file(record["checkpoint_path"]) != record["expected_sha256"]:
                raise ValueError("final checkpoint changed after plan validation")
            current = _artifact(artifacts, record["dataset"])
            evaluation = evaluate_checkpoint(
                record["checkpoint_path"], current.manifest_path, current.cache_root, record["dataset"],
                record["output_dir"], split="test", batch_size=batch_size, device=str(selected_device),
                store=prepared[record["dataset"]],
            )
            assert_same_evaluation_sets(record["set_signature"], evaluation["result"]["set_signature"])
            if evaluation["result"]["checkpoint_id"] != record["checkpoint_id"] or sha256_file(record["checkpoint_path"]) != record["expected_sha256"]:
                raise ValueError("final checkpoint changed during evaluation")
            summary["evaluations"].append({"target": record, **evaluation})
            summary["test_evaluated"] = True
            _atomic_json(summary, summary_path)
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        _atomic_json(summary, summary_path)
    return summary


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
            if "training_sets" in provenance:
                for split in ("train", "validation"):
                    assert_same_evaluation_sets(
                        provenance["training_sets"]["msp_podcast"][split],
                        evaluation_set_signature(store.manifest_path, "msp_podcast", split),
                    )
            elif provenance["evaluation_sets"]["msp_podcast"]["manifest_sha256"] != manifest_hash:
                # Old summaries identify the unchanged manifest here; no test scores are read.
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
    monitoring_config: TrainingMonitoringConfig | None = None,
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
    training_sets = {"msp_podcast": {
        "train": evaluation_set_signature(artifact.manifest_path, "msp_podcast", "train"),
        "validation": validation_signature,
    }}
    exclusion = bundle_msp_exclusion_contract(artifact, output)
    duplicates = bundle_msp_duplicate_provenance(artifact, output)
    rows = []
    runs = []
    summary_path = output / "comparison_summary.json"
    monitoring_kwargs = {"monitoring_config": monitoring_config} if monitoring_config is not None else {}
    for seed in seeds:
        baseline = baselines[seed]["training"]
        weighted = train_decoder(
            artifact.manifest_path, artifact.cache_root, "msp_podcast", output / f"seed-{seed}" / "balanced",
            replace(template, seed=seed, class_weighting="balanced"), training_stage="msp_train",
            store=store, **monitoring_kwargs,
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
        weighted_path = Path(weighted["best_checkpoint"])
        weighted_payload = load_decoder_checkpoint(weighted_path)
        runs.append({
            "seed": seed, "baseline": baselines[seed], "weighted": weighted, "validation_deltas": deltas,
            "provenance": {"training_sets": training_sets, "weighted_checkpoint": {
                "path": str(weighted_path), "sha256": sha256_file(weighted_path),
                "id": weighted_payload["checkpoint_id"], "cache_id": weighted_payload["cache_id"],
            }},
        })
        summary = {
            "dataset": "msp_podcast", "selection_split": "validation", "test_evaluated": False,
            "requested_seeds": list(seeds), "completed_seeds": [run["seed"] for run in runs],
            "base_config": asdict(template), "loss_config": loss_config,
            "monitoring_config": asdict(monitoring_config) if monitoring_config is not None else None,
            "cache_id": store.meta["cache_id"], "validation_signature": validation_signature,
            "training_sets": training_sets,
            "exclusion_contract_artifact": exclusion, "duplicate_provenance_artifact": duplicates,
            "rows": rows, "runs": runs, "summary_path": str(summary_path),
            "seconds": perf_counter() - started,
        }
        _atomic_json(summary, summary_path)
        score_deltas = {key: value for key, value in deltas.items() if key != "loss"}
        print(f"[MSP comparison seed={seed}] validation score changes: {score_deltas}", flush=True)
    return summary


def run_official_study(
    artifact, output_dir, official_head, parity_report, *, seeds=(42, 43, 44), config=None,
    monitoring_config=None, diagnostics_config=None,
):
    """Train independent D seeds from one official head without evaluating test data."""
    from .official import require_parity_report
    from .training import train_official_decoder

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("D seeds must be nonempty and unique")
    template = config or TrainingConfig(epochs=10, weight_decay=0)
    diagnostics_config = diagnostics_config or OfficialTrainingDiagnosticsConfig()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("official study output is not empty")
    store = ShardedFeatureStore(artifact.cache_root, artifact.manifest_path)
    require_parity_report(official_head, parity_report, store.meta)
    summary = {"condition_C": {"baseline": True, "official_snapshot": official_head.provenance,
                                "decision_rule": "softmax_official9_then_argmax_official9"},
               "requested_seeds": list(seeds), "runs": [], "test_evaluated": False,
               "selection_split": "hcudb1/validation", "summary_path": str(output / "official_study_summary.json"),
               "diagnostics_config": asdict(diagnostics_config),
               "diagnostics_aggregate": aggregate_official_training_diagnostics(
                   [], requested_seed_count=len(seeds),
               )}
    for seed in seeds:
        training = train_official_decoder(artifact.manifest_path, artifact.cache_root, output / f"seed-{seed}",
                                         official_head, parity_report, config=replace(template, seed=seed),
                                         monitoring_config=monitoring_config,
                                         diagnostics_config=diagnostics_config, store=store)
        diagnostic = training["diagnostics"]
        summary["runs"].append({
            "seed": seed,
            "training": training,
            "best": {"path": training["best_checkpoint"], "sha256": sha256_file(training["best_checkpoint"])},
            "diagnostics": {
                "path": diagnostic["artifacts"]["training_diagnostics_json"],
                "judgement": diagnostic["judgement"],
                "final_gap": diagnostic["gap_summary"]["final"],
                "best_epoch_gap": diagnostic["gap_summary"]["best_epoch"],
                "maximum_gap": diagnostic["gap_summary"]["maximum"],
            },
        })
        summary["diagnostics_aggregate"] = aggregate_official_training_diagnostics(
            [run["training"]["diagnostics"] for run in summary["runs"]],
            requested_seed_count=len(seeds),
        )
        _atomic_json(summary, Path(summary["summary_path"]))
    return summary


def _require_empty_official_evaluation_output(output_dir):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"official evaluation output is not empty: {output}")
    return output


def _prepare_official_evaluation_context(artifacts, official_head, parity_report, device):
    """Validate the shared head/cache/test identities used by C and D evaluation."""
    from .official import require_parity_report, state_sha256
    from .model import OfficialHeadModel

    if set(artifacts) != set(EVALUATION_DATASETS):
        raise ValueError("official evaluation requires MSP and HCUDB artifacts")
    prepared = {}
    datasets = {}
    for dataset in EVALUATION_DATASETS:
        artifact = artifacts[dataset]
        store = ShardedFeatureStore(artifact.cache_root, artifact.manifest_path)
        require_parity_report(official_head, parity_report, store.meta)
        test_set = evaluation_set_signature(artifact.manifest_path, dataset, "test")
        prepared[dataset] = store
        datasets[dataset] = {
            "cache_id": store.meta["cache_id"],
            "cache_manifest_sha256": store.meta["manifest_sha256"],
            "test_set": test_set,
        }
    head_sha256 = state_sha256(OfficialHeadModel(official_head).state_dict())
    signature = {
        "head": {"sha256": head_sha256, "official_snapshot": official_head.provenance},
        "datasets": datasets,
    }
    return prepared, signature, str(resolve_device(device))


def _validate_c_result(result, dataset, signature):
    expected = signature["datasets"][dataset]
    if result.get("condition") != "C" or not result.get("baseline"):
        raise ValueError(f"saved C evaluation is not a C baseline: {dataset}")
    if result.get("dataset") != dataset or result.get("split") != "test":
        raise ValueError(f"saved C evaluation dataset/split mismatch: {dataset}")
    if result.get("head_sha256") != signature["head"]["sha256"]:
        raise ValueError("saved C head differs from the current official head")
    if result.get("official_snapshot") != signature["head"]["official_snapshot"]:
        raise ValueError("saved C snapshot differs from the current official head")
    if result.get("cache_id") != expected["cache_id"]:
        raise ValueError(f"saved C cache differs from the current cache: {dataset}")
    assert_same_evaluation_sets(expected["test_set"], result["set_signature"])


def _load_official_c_summary(c_summary_path, signature):
    path = Path(c_summary_path)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid C evaluation summary: {path}") from exc
    if (summary.get("summary_schema_version") != "ser_official_c_evaluation_summary_v1"
            or summary.get("status") != "complete" or not summary.get("test_evaluated")):
        raise ValueError("C evaluation summary is not complete")
    saved_signature = summary.get("signature", {})
    if saved_signature.get("head") != signature["head"]:
        raise ValueError("saved C head differs from the current official head")
    if set(saved_signature.get("datasets", {})) != set(EVALUATION_DATASETS):
        raise ValueError("saved C summary does not contain both evaluation datasets")
    for dataset in EVALUATION_DATASETS:
        saved = saved_signature["datasets"][dataset]
        current = signature["datasets"][dataset]
        if (saved.get("cache_id") != current["cache_id"]
                or saved.get("cache_manifest_sha256") != current["cache_manifest_sha256"]):
            raise ValueError(f"saved C cache differs from the current cache: {dataset}")
        assert_same_evaluation_sets(saved.get("test_set", {}), current["test_set"])
    c_results = {}
    for evaluation in summary.get("evaluations", []):
        result = evaluation.get("result", {})
        dataset = result.get("dataset")
        if dataset in c_results:
            raise ValueError(f"saved C summary has duplicate evaluations: {dataset}")
        if dataset in EVALUATION_DATASETS:
            _validate_c_result(result, dataset, signature)
            c_results[dataset] = result
    if set(c_results) != set(EVALUATION_DATASETS) or len(summary.get("evaluations", [])) != len(EVALUATION_DATASETS):
        raise ValueError("saved C summary must contain exactly one C evaluation per dataset")
    return summary, c_results


def _freeze_official_d_checkpoints(d_checkpoints, official_head, hcudb_meta):
    from .checkpoints import load_official_checkpoint
    if not d_checkpoints:
        raise ValueError("D evaluation requires fixed D checkpoints")
    frozen = []
    for seed, record in sorted(d_checkpoints.items(), key=lambda item: int(item[0])):
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("final D checkpoint hash mismatch")
        payload = load_official_checkpoint(record["path"], official_head)
        if payload["selection"] != "best_validation" or payload["seed"] != int(seed):
            raise ValueError("final D must be the selected validation best for its seed")
        if (payload["signature"]["cache_id"] != hcudb_meta["cache_id"]
                or payload["signature"]["cache_manifest_sha256"] != hcudb_meta["manifest_sha256"]):
            raise ValueError("D training manifest/cache differs from final HCUDB artifacts")
        frozen.append({"seed": int(seed), "path": str(record["path"]), "sha256": record["sha256"], "checkpoint_id": payload["checkpoint_id"]})
    return frozen


def _evaluate_official_c_dataset(dataset, artifact, output, official_head, parity_report, store, signature, *, batch_size, device):
    from .training import evaluate_official

    evaluation = evaluate_official(
        artifact.manifest_path, artifact.cache_root, dataset, output,
        official_head, parity_report, device=device, batch_size=batch_size, store=store,
    )
    _validate_c_result(evaluation["result"], dataset, signature)
    return evaluation


def _evaluate_official_d_dataset(dataset, artifact, output, official_head, parity_report, store, frozen, c_result, *, batch_size, device):
    from .evaluation import assert_comparable_results
    from .training import evaluate_official

    evaluations = []
    results = []
    for record in frozen:
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("D checkpoint changed after freezing final plan")
        evaluation = evaluate_official(
            artifact.manifest_path, artifact.cache_root, dataset,
            output / f"D-seed-{record['seed']}", official_head, parity_report,
            checkpoint_path=record["path"], device=device, batch_size=batch_size, store=store,
        )
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("D checkpoint changed during final evaluation")
        result = evaluation["result"]
        if result.get("cache_id") != store.meta["cache_id"]:
            raise ValueError(f"D evaluation cache mismatch: {dataset}")
        assert_comparable_results(c_result, result)
        evaluations.append(evaluation)
        results.append(result)
    return evaluations, results


def _official_comparisons(c_result, d_results):
    import numpy as np

    comparisons = {}
    for metric in ("uar", "macro_f1", "accuracy", "loss"):
        baseline = c_result["metrics_target6"][metric]
        values = [result["metrics_target6"][metric] for result in d_results]
        comparisons[metric] = {
            "C": baseline,
            "D_by_seed": {str(result["seed"]): value for result, value in zip(d_results, values)},
            "D_mean": float(np.mean(values)),
            "D_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "mean_delta_from_C": float(np.mean(values) - baseline),
            "delta_by_seed": {str(result["seed"]): value - baseline for result, value in zip(d_results, values)},
        }
    return comparisons


def run_official_c_evaluations(artifacts, output_dir, official_head, parity_report, *, batch_size=8, device="auto"):
    """Evaluate C once on MSP Test1 and HCUDB Test and save its comparison signature."""
    output = _require_empty_official_evaluation_output(output_dir)
    prepared, signature, selected_device = _prepare_official_evaluation_context(
        artifacts, official_head, parity_report, device,
    )
    summary_path = output / "c_evaluation_summary.json"
    summary = {
        "summary_schema_version": "ser_official_c_evaluation_summary_v1",
        "status": "running",
        "condition": "C",
        "signature": signature,
        "plan": {"signature": signature, "batch_size": batch_size, "device": selected_device},
        "evaluations": [],
        "test_evaluated": False,
        "summary_path": str(summary_path),
    }
    try:
        for dataset in EVALUATION_DATASETS:
            evaluation = _evaluate_official_c_dataset(
                dataset, artifacts[dataset], output / dataset / "C", official_head, parity_report,
                prepared[dataset], signature, batch_size=batch_size, device=device,
            )
            summary["evaluations"].append(evaluation)
            summary["test_evaluated"] = True
            _atomic_json(summary, summary_path)
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        _atomic_json(summary, summary_path)
    return summary


def run_official_d_evaluations(
    artifacts, d_checkpoints, output_dir, official_head, parity_report, c_summary_path,
    *, batch_size=8, device="auto",
):
    """Evaluate only D checkpoints and compare them with a verified saved C summary."""
    output = _require_empty_official_evaluation_output(output_dir)
    prepared, signature, selected_device = _prepare_official_evaluation_context(
        artifacts, official_head, parity_report, device,
    )
    _, c_results = _load_official_c_summary(c_summary_path, signature)
    frozen = _freeze_official_d_checkpoints(d_checkpoints, official_head, prepared["hcudb1"].meta)
    plan = {
        "C": {"summary_path": str(c_summary_path), "signature": signature},
        "D": frozen,
        "batch_size": batch_size,
        "device": selected_device,
    }
    summary_path = output / "d_evaluation_summary.json"
    summary = {
        "summary_schema_version": "ser_official_d_evaluation_summary_v1",
        "status": "running",
        "plan": plan,
        "c_summary_path": str(c_summary_path),
        "evaluations": [],
        "comparisons": {},
        "test_evaluated": False,
        "summary_path": str(summary_path),
    }
    try:
        for dataset in EVALUATION_DATASETS:
            evaluations, d_results = _evaluate_official_d_dataset(
                dataset, artifacts[dataset], output / dataset, official_head, parity_report,
                prepared[dataset], frozen, c_results[dataset], batch_size=batch_size, device=device,
            )
            summary["evaluations"].extend(evaluations)
            summary["comparisons"][dataset] = _official_comparisons(c_results[dataset], d_results)
            summary["test_evaluated"] = True
            _atomic_json(summary, summary_path)
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        _atomic_json(summary, summary_path)
    return summary


def run_official_final_evaluations(artifacts, d_checkpoints, output_dir, official_head, parity_report, *, batch_size=8, device="auto"):
    """Compatibility API that evaluates C and D together in the historical output format."""
    output = _require_empty_official_evaluation_output(output_dir)
    prepared, signature, selected_device = _prepare_official_evaluation_context(
        artifacts, official_head, parity_report, device,
    )
    frozen = _freeze_official_d_checkpoints(d_checkpoints, official_head, prepared["hcudb1"].meta)
    sets = {dataset: signature["datasets"][dataset]["test_set"] for dataset in EVALUATION_DATASETS}
    plan = {"C": {"baseline": True, "head_sha256": signature["head"]["sha256"],
                  "snapshot": official_head.provenance},
            "D": frozen, "sets": sets, "batch_size": batch_size, "device": selected_device}
    _atomic_json(plan, output / "final_evaluation_plan.json")
    summary = {"status": "running", "plan": plan, "evaluations": [], "comparisons": {}, "test_evaluated": False}
    try:
        for dataset in EVALUATION_DATASETS:
            artifact = artifacts[dataset]
            c = _evaluate_official_c_dataset(
                dataset, artifact, output / dataset / "C", official_head, parity_report,
                prepared[dataset], signature, batch_size=batch_size, device=device,
            )
            summary["evaluations"].append(c)
            evaluations, d_results = _evaluate_official_d_dataset(
                dataset, artifact, output / dataset, official_head, parity_report,
                prepared[dataset], frozen, c["result"], batch_size=batch_size, device=device,
            )
            summary["evaluations"].extend(evaluations)
            summary["comparisons"][dataset] = _official_comparisons(c["result"], d_results)
            summary["test_evaluated"] = True
            _atomic_json(summary, output / "final_evaluation_summary.json")
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        _atomic_json(summary, output / "final_evaluation_summary.json")
    return summary


__all__ = [
    "DatasetArtifacts",
    "FinalEvaluationTarget",
    "EVALUATION_DATASETS",
    "STUDY_SEEDS",
    "bundle_msp_exclusion_contract",
    "bundle_msp_duplicate_provenance",
    "require_formal_epochs",
    "prepare_study_stores",
    "summarize_study",
    "run_msp_hcudb_study",
    "run_transfer_study",
    "run_final_evaluations",
    "load_msp_comparison_baselines",
    "run_msp_loss_comparison",
    "run_official_c_evaluations",
    "run_official_d_evaluations",
    "run_official_final_evaluations",
    "run_official_study",
]
