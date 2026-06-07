"""Train a VAD regressor on cached emotion2vec features."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import optim

try:
    from .data import (
        attach_cache_paths,
        build_vad_dataloader,
        ensure_feature_cache,
        load_vad_csv,
        split_vad_records,
    )
    from .loss import ccc_loss, vad_ccc_loss
    from .model import Emotion2VecVADRegressor, VAD_OUTPUT_NAMES
except ImportError:
    from data import (
        attach_cache_paths,
        build_vad_dataloader,
        ensure_feature_cache,
        load_vad_csv,
        split_vad_records,
    )
    from loss import ccc_loss, vad_ccc_loss
    from model import Emotion2VecVADRegressor, VAD_OUTPUT_NAMES


logger = logging.getLogger("train_vad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an emotion2vec feature head for 0..1 VAD regression."
    )
    parser.add_argument("--csv", required=True, help="CSV with file_path and VAD columns.")
    parser.add_argument("--audio-dir", default=None, help="Base directory for relative file_path values.")
    parser.add_argument("--cache-dir", required=True, help="Directory containing or receiving .npy features.")
    parser.add_argument("--output-dir", default="outputs/vad", help="Directory for checkpoints and metrics.")
    parser.add_argument("--extractor", default=None, help="Optional module:function used when cache is missing.")
    parser.add_argument("--force-cache", action="store_true", help="Regenerate cache files with --extractor.")
    parser.add_argument("--split-mode", default="auto", choices=("auto", "split", "session", "random"))
    parser.add_argument("--test-session", default=None, help="Held-out session for --split-mode session.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def load_extractor(spec: Optional[str]) -> Optional[Callable[[str], object]]:
    """Load an optional feature extractor from module:function."""
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError("--extractor must be in module:function format.")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    extractor = getattr(module, func_name)
    if not callable(extractor):
        raise TypeError(f"{spec} is not callable.")
    return extractor


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(value)


def prepare_records(args: argparse.Namespace):
    records = load_vad_csv(args.csv, audio_dir=args.audio_dir)
    records = attach_cache_paths(records, args.cache_dir)
    missing = [record for record in records if not Path(str(record["cache_path"])).exists()]

    if missing or args.force_cache:
        extractor = load_extractor(args.extractor)
        if extractor is None:
            sample = missing[0]["cache_path"] if missing else records[0]["cache_path"]
            raise FileNotFoundError(
                "Feature cache is incomplete. Provide --extractor module:function "
                f"or create cache files first. Example missing path: {sample}"
            )
        records = ensure_feature_cache(
            records,
            extractor=extractor,
            cache_dir=None,
            force=args.force_cache,
        )

    return records


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        net_input = batch["net_input"]
        feats = net_input["feats"].to(device)
        padding_mask = net_input["padding_mask"].to(device)
        targets = batch["vad_labels"].to(device)
        target_mask = batch["vad_mask"].to(device)

        optimizer.zero_grad()
        pred = model(feats, padding_mask)
        loss = vad_ccc_loss(pred, targets, target_mask)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Optional[Dict[str, object]]:
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds = []
    targets = []
    masks = []

    for batch in loader:
        net_input = batch["net_input"]
        feats = net_input["feats"].to(device)
        padding_mask = net_input["padding_mask"].to(device)
        target = batch["vad_labels"].to(device)
        target_mask = batch["vad_mask"].to(device)
        pred = model(feats, padding_mask)

        loss = vad_ccc_loss(pred, target, target_mask)
        total_loss += float(loss.detach().cpu())
        n_batches += 1
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
        masks.append(target_mask.detach().cpu())

    if n_batches == 0:
        return None

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    mask_all = torch.cat(masks, dim=0).bool()

    mae = {}
    ccc = {}
    for index, name in enumerate(VAD_OUTPUT_NAMES):
        valid = mask_all[:, index]
        if valid.any():
            mae[name] = float((pred_all[valid, index] - target_all[valid, index]).abs().mean())
            ccc[name] = float(1.0 - ccc_loss(pred_all[valid, index], target_all[valid, index]))
        else:
            mae[name] = None
            ccc[name] = None

    return {
        "loss": total_loss / n_batches,
        "mae": mae,
        "ccc": ccc,
        "n_samples": int(pred_all.size(0)),
    }


def make_loader(records: Iterable[dict], args: argparse.Namespace, shuffle: bool):
    records = list(records)
    if not records:
        return None
    return build_vad_dataloader(
        records,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = prepare_records(args)
    splits = split_vad_records(
        records,
        mode=args.split_mode,
        test_session=args.test_session,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if not splits["train"]:
        raise ValueError("The selected split produced no training rows.")

    train_loader = make_loader(splits["train"], args, shuffle=True)
    val_loader = make_loader(splits["val"], args, shuffle=False)
    test_loader = make_loader(splits["test"], args, shuffle=False)

    device = resolve_device(args.device)
    model = Emotion2VecVADRegressor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_score = float("inf")
    best_path = output_dir / "best_vad_regressor.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        selection_loss = val_metrics["loss"] if val_metrics is not None else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%s",
            epoch,
            train_loss,
            "none" if val_metrics is None else f"{val_metrics['loss']:.4f}",
        )
        if selection_loss < best_score:
            best_score = selection_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "output_names": VAD_OUTPUT_NAMES,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "best_selection_loss": best_score,
        "history": history,
        "test": evaluate(model, test_loader, device),
        "output_names": list(VAD_OUTPUT_NAMES),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    logger.info("saved checkpoint: %s", best_path)
    logger.info("saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
