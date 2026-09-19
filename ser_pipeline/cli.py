"""Command line entry points for manifest, feature, and study operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import LABEL_PROFILES, SUPPORTED_DATASETS
from .duplicates import generate_msp_audio_duplicate_exclusion_contract
from .manifest import (
    audit_dataset,
    build_manifest,
    generate_msp_audio_duplicate_audit,
    generate_msp_missing_audio_exclusion_contract,
    validate_manifest,
    load_manifest,
)
from .readers import resolved_dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ser-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-data", help="Audit metadata, labels, splits, and audio availability")
    audit.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--label-profile", choices=LABEL_PROFILES, default="ab4")

    exclusions = subparsers.add_parser(
        "generate-msp-exclusion-contract",
        help="Generate msp_missing_audio_exclusions_v1 from currently missing eligible audio",
    )
    exclusions.add_argument("--root", type=Path, required=True)
    exclusions.add_argument("--output", type=Path, required=True)
    exclusions.add_argument("--label-profile", choices=LABEL_PROFILES, default="ab4")

    duplicate_audit = subparsers.add_parser(
        "audit-msp-audio-duplicates",
        help="Generate msp_audio_duplicate_audit_v1 JSON and candidate CSV",
    )
    duplicate_audit.add_argument("--root", type=Path, required=True)
    duplicate_audit.add_argument("--audit-output", type=Path, required=True)
    duplicate_audit.add_argument("--candidates-csv-output", type=Path, required=True)
    duplicate_audit.add_argument("--approved-missing-exclusion-contract", type=Path, required=True)
    duplicate_audit.add_argument("--expected-missing-exclusion-sha256", required=True)
    duplicate_audit.add_argument("--label-profile", choices=LABEL_PROFILES, default="ab4")

    duplicate_exclusions = subparsers.add_parser(
        "generate-msp-duplicate-exclusion-contract",
        help="Generate msp_audio_duplicate_exclusions_v1 from explicitly approved candidate IDs",
    )
    duplicate_exclusions.add_argument("--audit", type=Path, required=True)
    duplicate_exclusions.add_argument("--approved-id", action="append", default=[])
    duplicate_exclusions.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build-manifest", help="Build a ser_manifest_v1 JSONL file")
    build.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allow-missing-audio", action="store_true")
    build.add_argument("--skip-excluded-audio-inspection", action="store_true")
    build.add_argument("--label-profile", choices=LABEL_PROFILES, default="ab4")
    build.add_argument(
        "--approved-exclusion-contract",
        type=Path,
        help="approved msp_missing_audio_exclusions_v1 JSON",
    )
    build.add_argument(
        "--expected-exclusion-sha256",
        help="approved normalized SHA-256 for the MSP exclusion contract",
    )
    build.add_argument("--duplicate-audit", type=Path, help="validated msp_audio_duplicate_audit_v1 JSON")
    build.add_argument(
        "--approved-duplicate-exclusion-contract",
        type=Path,
        help="approved msp_audio_duplicate_exclusions_v1 JSON",
    )
    build.add_argument(
        "--expected-duplicate-exclusion-sha256",
        help="approved normalized SHA-256 for the MSP duplicate exclusion contract",
    )

    validate = subparsers.add_parser("validate-manifest", help="Validate a ser_manifest_v1 JSONL file")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--root", type=Path, help="optionally recompute included audio metadata and SHA-256")

    extract = subparsers.add_parser("extract-features", help="Build or resume a sharded final-layer feature cache")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--audio-root", type=Path, required=True)
    extract.add_argument("--cache-root", type=Path, required=True)
    extract.add_argument("--user-dir", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--layer", default="final")
    extract.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    extract.add_argument("--max-shard-frames", type=int, default=65536)

    train = subparsers.add_parser("train-decoder", help="Train MSP parent or continue an HCUDB child")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--dataset", choices=("msp_podcast", "hcudb1"), required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--training-stage", choices=("msp_train", "hcudb_continue"), required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--parent-checkpoint", type=Path)
    train.add_argument("--resume-checkpoint", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a decoder checkpoint on a cached split")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--cache-root", type=Path, required=True)
    evaluate.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    benchmark = subparsers.add_parser("benchmark-audio", help="Benchmark one real audio extraction")
    benchmark.add_argument("--audio", type=Path, required=True)
    benchmark.add_argument("--user-dir", type=Path, required=True)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    verify = subparsers.add_parser("verify-official", help="Verify official logits/scores and benchmark one waveform")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--audio", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")

    large = subparsers.add_parser("extract-large", help="Extract shared B/C/D Large features after parity verification")
    large.add_argument("--snapshot", type=Path, required=True)
    large.add_argument("--parity-report", type=Path, required=True)
    large.add_argument("--manifest", type=Path, required=True)
    large.add_argument("--audio-root", type=Path, required=True)
    large.add_argument("--cache-root", type=Path, required=True)
    large.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    large.add_argument("--max-shard-frames", type=int, default=65536)
    large.add_argument("--preflight-report", type=Path)
    large.add_argument("--smoke-report", type=Path)

    official_eval = subparsers.add_parser("evaluate-official", help="Evaluate frozen C or a saved D head")
    for name in ("snapshot", "parity-report", "manifest", "cache-root", "output-dir"):
        official_eval.add_argument(f"--{name}", type=Path, required=True)
    official_eval.add_argument("--checkpoint", type=Path)
    official_eval.add_argument("--dataset", choices=("msp_podcast", "hcudb1"), required=True)
    official_eval.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    official_eval.add_argument("--batch-size", type=int, default=8)
    official_eval.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for command in ("train-official", "study-official"):
        trainer = subparsers.add_parser(command, help="Train D on HCUDB train/validation only")
        for name in ("snapshot", "parity-report", "manifest", "cache-root", "output-dir"):
            trainer.add_argument(f"--{name}", type=Path, required=True)
        trainer.add_argument("--epochs", type=int, default=10)
        trainer.add_argument("--batch-size", type=int, default=8)
        trainer.add_argument("--learning-rate", type=float, default=0.001)
        trainer.add_argument("--weight-decay", type=float, default=0)
        trainer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if command == "train-official":
            trainer.add_argument("--seed", type=int, default=42)
            trainer.add_argument("--resume-checkpoint", type=Path)
        else:
            trainer.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    final = subparsers.add_parser("compare-official", help="Run separate final C/D test comparison from frozen D hashes")
    for name in ("snapshot", "parity-report", "output-dir", "study-summary", "msp-manifest", "msp-cache-root", "hcudb-manifest", "hcudb-cache-root"):
        final.add_argument(f"--{name}", type=Path, required=True)
    final.add_argument("--batch-size", type=int, default=8)
    final.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-official":
        from .audio import load_audio_16k_mono
        from .cache import _atomic_json
        from .official import OfficialEmotion2vecEncoder
        encoder = OfficialEmotion2vecEncoder(args.snapshot, device=args.device)
        waveform = load_audio_16k_mono(args.audio)
        result = encoder.verify_parity(waveform)
        _atomic_json(result, args.output)
    elif args.command == "extract-large":
        from .cache import _atomic_json, cleanup_uncommitted_cache_fragments
        from .official import OfficialEmotion2vecEncoder, require_parity_report
        from .features import extract_feature_cache
        from .preflight import preflight_feature_extraction, smoke_test_feature_extraction

        rows = load_manifest(args.manifest)
        datasets = {str(row["dataset"]) for row in rows}
        if len(datasets) != 1:
            raise ValueError("extract-large requires a single-dataset manifest")
        dataset = next(iter(datasets))
        preflight_path = args.preflight_report or args.cache_root.parent / f"{args.cache_root.name}_preflight.json"
        print(f"[PRECHECK] {dataset} start")
        preflight = preflight_feature_extraction(
            args.manifest,
            args.audio_root,
            args.cache_root,
            args.parity_report,
            dataset=dataset,
            expected_dim=1024,
            max_shard_frames=args.max_shard_frames,
            capacity_path=args.cache_root,
            report_path=preflight_path,
        )
        dataset_preflight = preflight["datasets"][dataset]
        resolved_audio_root = Path(dataset_preflight["resolved_audio_root"])
        print(f"[PRECHECK] {dataset} complete")
        encoder = OfficialEmotion2vecEncoder(args.snapshot, device=args.device)
        report = require_parity_report(encoder.head, args.parity_report)
        if report["extraction"] != encoder.provenance or report["device"] != str(encoder.device):
            raise ValueError("re-run parity for the current extraction environment/device")
        print(f"[SMOKE] {dataset} start")
        smoke = smoke_test_feature_extraction(
            args.manifest,
            resolved_audio_root,
            args.cache_root,
            encoder,
            dataset=dataset,
            expected_dim=1024,
            max_shard_frames=args.max_shard_frames,
        )
        if args.smoke_report is not None:
            _atomic_json(smoke, args.smoke_report)
        print(f"[SMOKE] {dataset} complete")
        removed = cleanup_uncommitted_cache_fragments(
            args.cache_root,
            dataset_preflight["resume"]["recoverable_fragments"],
        )
        print(f"[RESUME] {dataset} committed={dataset_preflight['resume']['committed_utterances']} "
              f"pending={dataset_preflight['resume']['pending_utterances']} recovered={len(removed)}")
        if dataset_preflight["resume"]["complete"]:
            extraction = dataset_preflight["resume"]
            extraction["action"] = "validated/skip"
            print(f"[EXTRACT] {dataset} validated/skip")
        else:
            print(f"[EXTRACT] {dataset} start")
            extraction = extract_feature_cache(
                args.manifest,
                resolved_audio_root,
                args.cache_root,
                encoder,
                expected_dim=1024,
                max_shard_frames=args.max_shard_frames,
            )
            print(f"[EXTRACT] {dataset} complete")
        result = {
            "status": "ok",
            "dataset": dataset,
            "preflight": preflight,
            "smoke": smoke,
            "recovered_fragments": removed,
            "extraction": extraction,
        }
    elif args.command == "evaluate-official":
        from .official import load_official_head
        from .training import evaluate_official
        result = evaluate_official(args.manifest, args.cache_root, args.dataset, args.output_dir,
                                   load_official_head(args.snapshot), args.parity_report,
                                   checkpoint_path=args.checkpoint, split=args.split,
                                   batch_size=args.batch_size, device=args.device)
    elif args.command in {"train-official", "study-official"}:
        from .official import load_official_head
        from .training import TrainingConfig, train_official_decoder
        from .study import DatasetArtifacts, run_official_study
        head = load_official_head(args.snapshot)
        config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                weight_decay=args.weight_decay, device=args.device, seed=getattr(args, "seed", 42))
        if args.command == "train-official":
            result = train_official_decoder(args.manifest, args.cache_root, args.output_dir, head, args.parity_report,
                                           config=config, resume_checkpoint=args.resume_checkpoint)
        else:
            result = run_official_study(DatasetArtifacts(args.manifest, args.cache_root), args.output_dir,
                                        head, args.parity_report, seeds=args.seeds, config=config)
    elif args.command == "compare-official":
        from .official import load_official_head
        from .study import DatasetArtifacts, run_official_final_evaluations
        summary = json.loads(args.study_summary.read_text(encoding="utf-8"))
        if len(summary["runs"]) != len(summary["requested_seeds"]) or {r["seed"] for r in summary["runs"]} != set(summary["requested_seeds"]):
            raise ValueError("all requested D seeds must finish before final comparison")
        artifacts = {"msp_podcast": DatasetArtifacts(args.msp_manifest, args.msp_cache_root),
                     "hcudb1": DatasetArtifacts(args.hcudb_manifest, args.hcudb_cache_root)}
        result = run_official_final_evaluations(artifacts, {r["seed"]: r["best"] for r in summary["runs"]},
                                               args.output_dir, load_official_head(args.snapshot), args.parity_report,
                                               batch_size=args.batch_size, device=args.device)
    elif args.command == "audit-data":
        result = audit_dataset(args.dataset, args.root, label_profile=args.label_profile)
    elif args.command == "generate-msp-exclusion-contract":
        result = generate_msp_missing_audio_exclusion_contract(
            args.root, args.output, label_profile=args.label_profile
        )
    elif args.command == "audit-msp-audio-duplicates":
        result = generate_msp_audio_duplicate_audit(
            args.root,
            args.audit_output,
            args.candidates_csv_output,
            approved_missing_audio_exclusion_contract=args.approved_missing_exclusion_contract,
            expected_missing_audio_exclusion_sha256=args.expected_missing_exclusion_sha256,
            label_profile=args.label_profile,
        )
    elif args.command == "generate-msp-duplicate-exclusion-contract":
        result = generate_msp_audio_duplicate_exclusion_contract(
            args.audit,
            args.approved_id,
            args.output,
        )
    elif args.command == "build-manifest":
        result = build_manifest(
            args.dataset,
            args.root,
            args.output,
            strict=not args.allow_missing_audio,
            inspect_excluded_audio=not args.skip_excluded_audio_inspection,
            approved_exclusion_contract=args.approved_exclusion_contract,
            expected_exclusion_sha256=args.expected_exclusion_sha256,
            duplicate_audit=args.duplicate_audit,
            approved_duplicate_exclusion_contract=args.approved_duplicate_exclusion_contract,
            expected_duplicate_exclusion_sha256=args.expected_duplicate_exclusion_sha256,
            label_profile=args.label_profile,
        )
    elif args.command == "validate-manifest":
        result = validate_manifest(args.manifest, audio_root=args.root)
    elif args.command == "extract-features":
        from .features import Emotion2vecEncoder, extract_feature_cache

        rows = load_manifest(args.manifest)
        datasets = {str(row["dataset"]) for row in rows}
        if len(datasets) != 1:
            raise ValueError("extract-features requires a single-dataset manifest")
        dataset = next(iter(datasets))
        audio_root = resolved_dataset_root(dataset, args.audio_root).resolve()
        validate_manifest(args.manifest, audio_root=audio_root, audio_root_resolved=True)
        encoder = Emotion2vecEncoder(
            args.user_dir,
            args.checkpoint,
            layer=args.layer,
            device=args.device,
        )
        result = extract_feature_cache(
            args.manifest,
            audio_root,
            args.cache_root,
            encoder,
            layer=args.layer,
            max_shard_frames=args.max_shard_frames,
            expected_dim=768,
        )
    elif args.command == "train-decoder":
        from .training import TrainingConfig, train_decoder

        config = TrainingConfig(
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        result = train_decoder(
            args.manifest,
            args.cache_root,
            args.dataset,
            args.output_dir,
            config,
            training_stage=args.training_stage,
            parent_checkpoint=args.parent_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
        )
    elif args.command == "evaluate":
        from .training import evaluate_checkpoint

        result = evaluate_checkpoint(
            args.checkpoint,
            args.manifest,
            args.cache_root,
            args.dataset,
            args.output_dir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "benchmark-audio":
        from .preflight import benchmark_audio_extraction, save_benchmark

        result = benchmark_audio_extraction(
            args.audio,
            args.user_dir,
            args.checkpoint,
            device=args.device,
        )
        save_benchmark(result, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
