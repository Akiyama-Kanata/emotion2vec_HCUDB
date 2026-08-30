"""独立したカテゴリ感情ヘッドと V/A/D 回帰ヘッドを同時学習する CLI。"""

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from vad_downstream.data import (
        EMOTION_CLASS_LABELS,
        EMOTION_CLASS_NAMES_JA,
        VADEmotionSpeechDataset,
        load_vad_emotion_dataset,
    )
    from vad_downstream.model import ParallelEmotionVADClassifier
    from vad_downstream.parallel_training import (
        evaluate,
        save_parallel_checkpoint,
        train_one_epoch,
    )
except ModuleNotFoundError:
    from data import (
        EMOTION_CLASS_LABELS,
        EMOTION_CLASS_NAMES_JA,
        VADEmotionSpeechDataset,
        load_vad_emotion_dataset,
    )
    from model import ParallelEmotionVADClassifier
    from parallel_training import evaluate, save_parallel_checkpoint, train_one_epoch


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train independent emotion and V/A/D heads from emotion2vec features."
    )
    parser.add_argument("--train-prefix", required=True)
    parser.add_argument("--valid-prefix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--class-labels", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lambda-vad", type=float, default=1.0)
    parser.add_argument("--lambda-emo", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    summary = train_from_args(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def train_from_args(args):
    _validate_args(args)
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    initial = (
        None
        if args.initial_checkpoint is None
        else torch.load(args.initial_checkpoint, map_location="cpu")
    )
    class_labels = _resolve_class_labels(args.class_labels, initial)
    train_data, train_loader = build_data_loader(
        args.train_prefix,
        class_labels,
        args.batch_size,
        args.min_length,
        args.max_length,
        True,
        args.seed,
    )
    valid_data = valid_loader = None
    if args.valid_prefix:
        valid_data, valid_loader = build_data_loader(
            args.valid_prefix,
            class_labels,
            args.batch_size,
            args.min_length,
            args.max_length,
            False,
            args.seed,
        )

    model = ParallelEmotionVADClassifier(
        num_classes=len(class_labels), hidden_dim=args.hidden_dim
    )
    if initial is not None:
        if initial.get("model_type") != "parallel_emotion_vad":
            raise ValueError("--initial-checkpoint must be a parallel_emotion_vad model")
        if int(initial.get("hidden_dim", args.hidden_dim)) != args.hidden_dim:
            raise ValueError("--hidden-dim must match the initial checkpoint")
        model.load_state_dict(initial["model_state_dict"])

    counts_array = train_data["vad_label_counts"]
    has_dominance = int(counts_array[2]) > 0
    if has_dominance:
        dominance_status = "trained"
        include_dominance = True
    elif initial is not None and initial.get("dominance_status") in {
        "trained",
        "retained_from_checkpoint",
    }:
        dominance_status = "retained_from_checkpoint"
        include_dominance = False
    else:
        dominance_status = "untrained"
        include_dominance = False
    if not include_dominance:
        for parameter in model.dominance_head.parameters():
            parameter.requires_grad = False

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.task_parameters(include_dominance=include_dominance),
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
        entry = {"epoch": epoch, "train": train_metrics}
        if valid_loader is not None:
            valid_metrics = evaluate(
                model,
                valid_loader,
                device,
                lambda_vad=args.lambda_vad,
                lambda_emo=args.lambda_emo,
                class_labels=class_labels,
            )
            entry["valid"] = valid_metrics
            if best is None or valid_metrics["loss"] < best["loss"]:
                best = {
                    "loss": valid_metrics["loss"],
                    "epoch": epoch,
                    "metrics": valid_metrics,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
        history.append(entry)

    if best is not None:
        model.load_state_dict(best["state"])
        saved_epoch, selection = best["epoch"], "best_valid_loss"
    else:
        saved_epoch, selection = args.epochs, "final"
    class_names = (
        list(EMOTION_CLASS_NAMES_JA)
        if class_labels == list(EMOTION_CLASS_LABELS)
        else list(class_labels)
    )
    metadata = {
        "train_prefix": args.train_prefix,
        "valid_prefix": args.valid_prefix,
        "history": history,
        "saved_epoch": saved_epoch,
        "selection": selection,
        "initial_checkpoint": args.initial_checkpoint,
    }
    model.cpu()
    counts = {
        name: int(count)
        for name, count in zip(("valence", "arousal", "dominance"), counts_array)
    }
    save_parallel_checkpoint(
        model,
        args.output,
        class_labels,
        counts,
        dominance_status,
        class_names_ja=class_names,
        lambda_vad=args.lambda_vad,
        lambda_emo=args.lambda_emo,
        metadata=metadata,
    )
    return {
        "output": str(args.output),
        "model_type": "parallel_emotion_vad",
        "class_labels": class_labels,
        "num_classes": len(class_labels),
        "vad_label_counts": counts,
        "supervised_dimensions": [name for name, count in counts.items() if count],
        "dominance_status": dominance_status,
        "saved_epoch": saved_epoch,
        "selection": selection,
        "train": history[-1]["train"],
        "valid": None if best is None else best["metrics"],
        "history": history,
        "device": str(device),
    }


def build_data_loader(prefix, class_labels, batch_size, min_length, max_length, shuffle, seed):
    data = load_vad_emotion_dataset(
        prefix,
        min_length=min_length,
        max_length=max_length,
        class_labels=class_labels,
    )
    if data["num"] == 0:
        raise ValueError("dataset has no samples after length filtering")
    dataset = VADEmotionSpeechDataset(
        data["feats"],
        data["sizes"],
        data["offsets"],
        data["vad_targets"],
        data["emotion_targets"],
        utt_ids=data["utt_ids"],
        emotion_labels=data["emotion_labels"],
        class_labels=data["class_labels"],
        vad_target_masks=data["vad_target_masks"],
    )
    generator = torch.Generator().manual_seed(seed)
    return data, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collator,
        generator=generator if shuffle else None,
    )


def _resolve_class_labels(cli_labels, initial):
    labels = list(cli_labels) if cli_labels is not None else None
    if initial is not None:
        checkpoint_labels = list(initial.get("class_labels", EMOTION_CLASS_LABELS))
        if labels is not None and labels != checkpoint_labels:
            raise ValueError("--class-labels must match the initial checkpoint")
        return checkpoint_labels
    return list(EMOTION_CLASS_LABELS) if labels is None else labels


def _validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        raise ValueError("epochs, batch-size, and hidden-dim must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay non-negative")
    if args.lambda_vad < 0 or args.lambda_emo < 0 or (
        args.lambda_vad == 0 and args.lambda_emo == 0
    ):
        raise ValueError("loss weights must be non-negative and not both zero")
    if args.class_labels is not None and (
        len(args.class_labels) < 2 or len(set(args.class_labels)) != len(args.class_labels)
    ):
        raise ValueError("--class-labels needs at least two unique labels")


def _resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return torch.device(value)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
