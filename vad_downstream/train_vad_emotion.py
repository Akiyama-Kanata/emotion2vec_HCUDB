"""推定 VAD を分類器入力に使うカテゴリ感情モデルを学習する CLI。"""

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from vad_downstream.data import (
        EMOTION_CLASS_LABELS,
        VADEmotionSpeechDataset,
        load_vad_emotion_dataset,
    )
    from vad_downstream.emotion_training import (
        copy_cpu_state_dict,
        evaluate,
        save_vad_emotion_checkpoint,
        train_one_epoch,
    )
    from vad_downstream.model import VADMediatedEmotionClassifier
except ModuleNotFoundError:
    from data import (
        EMOTION_CLASS_LABELS,
        VADEmotionSpeechDataset,
        load_vad_emotion_dataset,
    )
    from emotion_training import (
        copy_cpu_state_dict,
        evaluate,
        save_vad_emotion_checkpoint,
        train_one_epoch,
    )
    from model import VADMediatedEmotionClassifier


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a VAD-mediated emotion classifier from precomputed "
            "emotion2vec features."
        )
    )
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--valid-prefix", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--target-dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--lambda-vad", type=float, default=1.0)
    parser.add_argument("--lambda-emo", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    summary = train_from_args(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def train_from_args(args):
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    train_data, train_loader = build_data_loader(
        prefix=args.train_prefix,
        batch_size=args.batch_size,
        min_length=args.min_length,
        max_length=args.max_length,
        shuffle=True,
        seed=args.seed,
        split_name="train",
    )
    validate_target_dim(train_data, args.target_dim, split_name="train")

    valid_data = None
    valid_loader = None
    if args.valid_prefix is not None:
        valid_data, valid_loader = build_data_loader(
            prefix=args.valid_prefix,
            batch_size=args.batch_size,
            min_length=args.min_length,
            max_length=args.max_length,
            shuffle=False,
            seed=args.seed,
            split_name="valid",
        )
        validate_target_dim(valid_data, args.target_dim, split_name="valid")

    model = VADMediatedEmotionClassifier(
        target_dim=args.target_dim,
        num_classes=train_data["num_classes"],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            lambda_vad=args.lambda_vad,
            lambda_emo=args.lambda_emo,
        )
        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
        }

        if valid_loader is not None:
            valid_metrics = evaluate(
                model,
                valid_loader,
                device,
                lambda_vad=args.lambda_vad,
                lambda_emo=args.lambda_emo,
                class_labels=train_data["class_labels"],
            )
            epoch_summary["valid"] = valid_metrics
            if best is None or valid_metrics["loss"] < best["loss"]:
                best = {
                    "epoch": epoch,
                    "loss": valid_metrics["loss"],
                    "metrics": valid_metrics,
                    "state_dict": copy_cpu_state_dict(model),
                }

        history.append(epoch_summary)

    if best is not None:
        model.load_state_dict(best["state_dict"])
        saved_epoch = int(best["epoch"])
        selection = "best_valid_loss"
        saved_valid_metrics = best["metrics"]
    else:
        saved_epoch = int(args.epochs)
        selection = "final"
        saved_valid_metrics = None

    last_train_metrics = history[-1]["train"]
    metadata = {
        "train_prefix": args.train_prefix,
        "valid_prefix": args.valid_prefix,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "target_dim": int(args.target_dim),
        "lambda_vad": float(args.lambda_vad),
        "lambda_emo": float(args.lambda_emo),
        "seed": int(args.seed),
        "min_length": int(args.min_length),
        "max_length": None if args.max_length is None else int(args.max_length),
        "train_num_samples": int(train_data["num"]),
        "valid_num_samples": None if valid_data is None else int(valid_data["num"]),
        "saved_epoch": saved_epoch,
        "selection": selection,
        "history": history,
    }

    model.to(torch.device("cpu"))
    save_vad_emotion_checkpoint(
        model=model,
        output_path=args.output,
        target_dim=args.target_dim,
        class_labels=train_data["class_labels"],
        class_names_ja=train_data["class_names_ja"],
        lambda_vad=args.lambda_vad,
        lambda_emo=args.lambda_emo,
        metadata=metadata,
    )

    return {
        "output": str(args.output),
        "target_dim": int(args.target_dim),
        "input_dim": int(model.input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_classes": int(train_data["num_classes"]),
        "class_labels": train_data["class_labels"],
        "class_names_ja": train_data["class_names_ja"],
        "device": str(device),
        "saved_epoch": saved_epoch,
        "selection": selection,
        "train": {
            "num_samples": int(train_data["num"]),
            "last": last_train_metrics,
        },
        "valid": saved_valid_metrics,
        "history": history,
    }


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be at least 1")
    if args.lambda_vad < 0.0:
        raise ValueError("--lambda-vad must be non-negative")
    if args.lambda_emo < 0.0:
        raise ValueError("--lambda-emo must be non-negative")
    if args.lambda_vad == 0.0 and args.lambda_emo == 0.0:
        raise ValueError("at least one loss weight must be positive")
    if args.min_length < 0:
        raise ValueError("--min-length must be non-negative")
    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("--max-length must be greater than or equal to --min-length")


def validate_target_dim(data, target_dim, split_name):
    if int(data["target_dim"]) != int(target_dim):
        raise ValueError(
            f"{split_name} target_dim ({int(data['target_dim'])}) does not match "
            f"requested target_dim ({int(target_dim)})"
        )
    # The legacy VAD-mediated comparison model requires a uniformly complete
    # target. Mixed/missing D supervision belongs to the parallel model CLI.
    if int(target_dim) == 3 and "vad_target_masks" in data:
        dominance_count = int(data["vad_target_masks"][:, 2].sum())
        if dominance_count != int(data["num"]):
            raise ValueError(
                f"{split_name} target_dim (VA with missing dominance) does not "
                "match requested target_dim (3)"
            )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is not available")
    return torch.device(device)


def build_data_loader(
    prefix,
    batch_size,
    min_length,
    max_length,
    shuffle,
    seed,
    split_name,
):
    data = load_vad_emotion_dataset(
        prefix,
        min_length=min_length,
        max_length=max_length,
        class_labels=EMOTION_CLASS_LABELS,
        masked_vad=False,
    )
    if data["num"] == 0:
        raise ValueError(f"{split_name} dataset has no samples after length filtering")

    dataset = VADEmotionSpeechDataset(
        data["feats"],
        data["sizes"],
        data["offsets"],
        data["vad_targets"],
        data["emotion_targets"],
        data["utt_ids"],
        data["emotion_labels"],
        class_labels=data["class_labels"],
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collator,
        generator=generator if shuffle else None,
    )
    return data, loader


if __name__ == "__main__":
    main()
