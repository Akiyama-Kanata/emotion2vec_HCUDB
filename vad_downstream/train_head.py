"""`<prefix>.npy/.lengths/.vad` から VAD 回帰ヘッドだけを学習する CLI。"""

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from vad_downstream.data import VADSpeechDataset, load_vad_dataset
    from vad_downstream.model import VADRegressionHead
    from vad_downstream.training import evaluate, save_head_checkpoint, train_one_epoch
except ModuleNotFoundError:
    from data import VADSpeechDataset, load_vad_dataset
    from model import VADRegressionHead
    from training import evaluate, save_head_checkpoint, train_one_epoch


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a VADRegressionHead from precomputed emotion2vec features."
    )
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--valid-prefix", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
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
        if train_data["target_dim"] != valid_data["target_dim"]:
            raise ValueError(
                f"train target_dim ({train_data['target_dim']}) does not match "
                f"valid target_dim ({valid_data['target_dim']})"
            )

    head = VADRegressionHead(
        target_dim=train_data["target_dim"],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(head, optimizer, train_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(train_loss),
        }

        if valid_loader is not None:
            valid_metrics = evaluate(head, valid_loader, device)
            epoch_summary["valid"] = valid_metrics
            if best is None or valid_metrics["mean_ccc"] > best["mean_ccc"]:
                best = {
                    "epoch": epoch,
                    "mean_ccc": valid_metrics["mean_ccc"],
                    "metrics": valid_metrics,
                    "state_dict": copy_cpu_state_dict(head),
                }

        history.append(epoch_summary)

    if best is not None:
        head.load_state_dict(best["state_dict"])
        saved_epoch = int(best["epoch"])
        selection = "best_valid_mean_ccc"
        saved_valid_metrics = best["metrics"]
    else:
        saved_epoch = int(args.epochs)
        selection = "final"
        saved_valid_metrics = None

    metadata = {
        "train_prefix": args.train_prefix,
        "valid_prefix": args.valid_prefix,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "seed": int(args.seed),
        "min_length": int(args.min_length),
        "max_length": None if args.max_length is None else int(args.max_length),
        "train_num_samples": int(train_data["num"]),
        "valid_num_samples": None if valid_data is None else int(valid_data["num"]),
        "saved_epoch": saved_epoch,
        "selection": selection,
        "history": history,
    }

    head.to(torch.device("cpu"))
    save_head_checkpoint(
        head=head,
        output_path=args.output,
        target_dim=train_data["target_dim"],
        metadata=metadata,
    )

    return {
        "output": str(args.output),
        "target_dim": int(train_data["target_dim"]),
        "input_dim": int(head.input_dim),
        "hidden_dim": int(args.hidden_dim),
        "device": str(device),
        "saved_epoch": saved_epoch,
        "selection": selection,
        "train": {
            "num_samples": int(train_data["num"]),
            "last_loss": float(history[-1]["train_loss"]),
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
    if args.min_length < 0:
        raise ValueError("--min-length must be non-negative")
    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("--max-length must be greater than or equal to --min-length")


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
    data = load_vad_dataset(
        prefix,
        min_length=min_length,
        max_length=max_length,
    )
    if data["num"] == 0:
        raise ValueError(f"{split_name} dataset has no samples after length filtering")

    dataset = VADSpeechDataset(
        data["feats"],
        data["sizes"],
        data["offsets"],
        data["targets"],
        data["utt_ids"],
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


def copy_cpu_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


if __name__ == "__main__":
    main()
