"""連結済み emotion2vec 特徴量から VAD 回帰ヘッドを学習・評価・保存する。"""

from pathlib import Path

import torch


def concordance_correlation_coefficient(prediction, target, eps=1e-8):
    """Compute CCC per VA/VAD dimension over a batch."""
    _validate_prediction_and_target(prediction, target)

    prediction_mean = prediction.mean(dim=0)
    target_mean = target.mean(dim=0)
    prediction_var = prediction.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    covariance = (
        (prediction - prediction_mean) * (target - target_mean)
    ).mean(dim=0)

    denominator = (
        prediction_var
        + target_var
        + torch.square(prediction_mean - target_mean)
        + eps
    )
    return 2.0 * covariance / denominator


def ccc_loss(prediction, target, eps=1e-8):
    """Return 1 - mean CCC for VA/VAD regression."""
    ccc = concordance_correlation_coefficient(prediction, target, eps=eps)
    return 1.0 - ccc.mean()


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    criterion=ccc_loss,
    input_key="feats",
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        net_input = batch["net_input"]
        if input_key not in net_input:
            raise ValueError(f"batch net_input does not contain '{input_key}'")

        model_input = net_input[input_key].to(device)
        padding_mask = net_input.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        prediction = model(model_input, padding_mask=padding_mask)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        num_batches += 1

    if num_batches == 0:
        raise ValueError("data_loader must yield at least one batch")

    return total_loss / num_batches


def evaluate(
    model,
    data_loader,
    device,
    criterion=ccc_loss,
    input_key="feats",
):
    model.eval()
    predictions = []
    targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            net_input = batch["net_input"]
            if input_key not in net_input:
                raise ValueError(f"batch net_input does not contain '{input_key}'")

            model_input = net_input[input_key].to(device)
            padding_mask = net_input.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            target = batch["target"].to(device)

            prediction = model(model_input, padding_mask=padding_mask)
            predictions.append(prediction.detach().cpu())
            targets.append(target.detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("data_loader must yield at least one batch")

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    loss = criterion(prediction, target)
    ccc = concordance_correlation_coefficient(prediction, target)

    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "valence_ccc": float(ccc[0].detach().cpu().item()),
        "arousal_ccc": float(ccc[1].detach().cpu().item()),
        "mean_ccc": float(ccc.mean().detach().cpu().item()),
        "num_samples": int(target.size(0)),
        "num_batches": int(num_batches),
    }
    if target.size(1) == 3:
        metrics["dominance_ccc"] = float(ccc[2].detach().cpu().item())
    return metrics


def save_head_checkpoint(head, output_path, target_dim, metadata=None):
    target_dim = int(target_dim)
    if target_dim not in (2, 3):
        raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")
    if hasattr(head, "target_dim") and int(head.target_dim) != target_dim:
        raise ValueError(
            f"head target_dim ({int(head.target_dim)}) does not match "
            f"target_dim ({target_dim})"
        )

    input_dim = int(getattr(head, "input_dim", head.pre_net.in_features))
    hidden_dim = int(getattr(head, "hidden_dim", head.pre_net.out_features))
    checkpoint = {
        "head_state_dict": head.state_dict(),
        "target_dim": target_dim,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "metadata": {} if metadata is None else metadata,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def _validate_prediction_and_target(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{prediction.shape} and {target.shape}"
        )
    if prediction.dim() != 2:
        raise ValueError(
            f"prediction and target must be 2D [B, D], got {prediction.shape}"
        )
    if prediction.size(1) not in (2, 3):
        raise ValueError(
            f"VA/VAD target dimension must be 2 or 3, got {prediction.size(1)}"
        )
