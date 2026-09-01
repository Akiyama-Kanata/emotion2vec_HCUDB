"""Command line entry points for manifest, feature, and study operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import SUPPORTED_DATASETS
from .duplicates import generate_msp_audio_duplicate_exclusion_contract
from .manifest import (
    audit_dataset,
    build_manifest,
    generate_msp_audio_duplicate_audit,
    generate_msp_missing_audio_exclusion_contract,
    validate_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ser-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-data", help="Audit metadata, labels, splits, and audio availability")
    audit.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    audit.add_argument("--root", type=Path, required=True)

    exclusions = subparsers.add_parser(
        "generate-msp-exclusion-contract",
        help="Generate msp_missing_audio_exclusions_v1 from currently missing eligible audio",
    )
    exclusions.add_argument("--root", type=Path, required=True)
    exclusions.add_argument("--output", type=Path, required=True)

    duplicate_audit = subparsers.add_parser(
        "audit-msp-audio-duplicates",
        help="Generate msp_audio_duplicate_audit_v1 JSON and candidate CSV",
    )
    duplicate_audit.add_argument("--root", type=Path, required=True)
    duplicate_audit.add_argument("--audit-output", type=Path, required=True)
    duplicate_audit.add_argument("--candidates-csv-output", type=Path, required=True)
    duplicate_audit.add_argument("--approved-missing-exclusion-contract", type=Path, required=True)
    duplicate_audit.add_argument("--expected-missing-exclusion-sha256", required=True)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-data":
        result = audit_dataset(args.dataset, args.root)
    elif args.command == "generate-msp-exclusion-contract":
        result = generate_msp_missing_audio_exclusion_contract(args.root, args.output)
    elif args.command == "audit-msp-audio-duplicates":
        result = generate_msp_audio_duplicate_audit(
            args.root,
            args.audit_output,
            args.candidates_csv_output,
            approved_missing_audio_exclusion_contract=args.approved_missing_exclusion_contract,
            expected_missing_audio_exclusion_sha256=args.expected_missing_exclusion_sha256,
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
        )
    elif args.command == "validate-manifest":
        result = validate_manifest(args.manifest, audio_root=args.root)
    elif args.command == "extract-features":
        from .features import Emotion2vecEncoder, extract_feature_cache

        encoder = Emotion2vecEncoder(
            args.user_dir,
            args.checkpoint,
            layer=args.layer,
            device=args.device,
        )
        result = extract_feature_cache(
            args.manifest,
            args.audio_root,
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
