from pathlib import Path

import torch
from torch import nn

try:
    from vad_downstream.emotion_training import classification_metrics
except ModuleNotFoundError:
    from emotion_training import classification_metrics


VAD_DIMENSIONS = ("valence", "arousal", "dominance")
DOMINANCE_STATUSES = {
    "trained",
    "untrained",
    "retained_from_checkpoint",
}


def compute_parallel_loss(
    output,
    vad_target,
    vad_target_mask,
    emotion_target,
    lambda_vad=1.0,
    lambda_emo=1.0,
    emotion_criterion=None,
):
    """Compute independent CE and per-dimension masked CCC losses."""
    _validate_batch(output, vad_target, vad_target_mask, emotion_target)
    if lambda_vad < 0 or lambda_emo < 0 or (lambda_vad == 0 and lambda_emo == 0):
        raise ValueError("loss weights must be non-negative and not both zero")
    if emotion_criterion is None:
        emotion_criterion = nn.CrossEntropyLoss()

    dimension_losses = {}
    included = []
    for index, name in enumerate(VAD_DIMENSIONS):
        mask = vad_target_mask[:, index]
        # D is deliberately skipped for a singleton subset because its batch CCC
        # is not a useful or stable training signal.
        if name == "dominance" and int(mask.sum().item()) < 2:
            dimension_losses[name] = None
            continue
        if int(mask.sum().item()) == 0:
            dimension_losses[name] = None
            continue
        ccc = _scalar_ccc(
            output["vad"][mask, index],
            vad_target[mask, index],
        )
        dimension_losses[name] = 1.0 - ccc
        included.append(dimension_losses[name])

    if not included:
        vad_loss = output["vad"].sum() * 0.0
    else:
        vad_loss = torch.stack(included).mean()
    emotion_loss = emotion_criterion(output["logits"], emotion_target)
    return {
        "loss": float(lambda_vad) * vad_loss + float(lambda_emo) * emotion_loss,
        "vad_loss": vad_loss,
        "emotion_loss": emotion_loss,
        "dimension_losses": dimension_losses,
        "dominance_loss_skipped": dimension_losses["dominance"] is None,
    }


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    lambda_vad=1.0,
    lambda_emo=1.0,
    input_key="feats",
):
    model.train()
    totals = {"loss": 0.0, "vad_loss": 0.0, "emotion_loss": 0.0}
    batches = 0
    dominance_batches = 0
    for batch in data_loader:
        model_input, padding_mask, vad_target, vad_mask, emotion_target = _prepare_batch(
            batch, device, input_key
        )
        optimizer.zero_grad()
        output = model(model_input, padding_mask=padding_mask)
        losses = compute_parallel_loss(
            output,
            vad_target,
            vad_mask,
            emotion_target,
            lambda_vad=lambda_vad,
            lambda_emo=lambda_emo,
        )
        losses["loss"].backward()
        if losses["dominance_loss_skipped"]:
            dominance_head = getattr(model, "dominance_head", None)
            if dominance_head is None and hasattr(model, "head"):
                dominance_head = getattr(model.head, "dominance_head", None)
            if dominance_head is not None:
                # CatBackward can materialize zero gradients for the unused D
                # slice. Clear them so AdamW weight decay also skips D exactly.
                for parameter in dominance_head.parameters():
                    parameter.grad = None
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu().item())
        dominance_batches += int(not losses["dominance_loss_skipped"])
        batches += 1
    if batches == 0:
        raise ValueError("data_loader must yield at least one batch")
    result = {key: value / batches for key, value in totals.items()}
    result.update(
        {"num_batches": batches, "dominance_loss_batches": dominance_batches}
    )
    return result


def evaluate(
    model,
    data_loader,
    device,
    lambda_vad=1.0,
    lambda_emo=1.0,
    input_key="feats",
    class_labels=None,
):
    model.eval()
    vad_predictions, vad_targets, masks, logits, emotion_targets = [], [], [], [], []
    batches = 0
    with torch.no_grad():
        for batch in data_loader:
            model_input, padding_mask, target, mask, emotion_target = _prepare_batch(
                batch, device, input_key
            )
            output = model(model_input, padding_mask=padding_mask)
            vad_predictions.append(output["vad"].cpu())
            vad_targets.append(target.cpu())
            masks.append(mask.cpu())
            logits.append(output["logits"].cpu())
            emotion_targets.append(emotion_target.cpu())
            batches += 1
    if batches == 0:
        raise ValueError("data_loader must yield at least one batch")

    prediction = torch.cat(vad_predictions)
    target = torch.cat(vad_targets)
    mask = torch.cat(masks)
    emotion_logits = torch.cat(logits)
    emotion_target = torch.cat(emotion_targets)
    losses = compute_parallel_loss(
        {"vad": prediction, "logits": emotion_logits},
        target,
        mask,
        emotion_target,
        lambda_vad=lambda_vad,
        lambda_emo=lambda_emo,
    )
    result = classification_metrics(
        emotion_logits, emotion_target, class_labels=class_labels
    )
    result.update(
        {
            "loss": float(losses["loss"].item()),
            "vad_loss": float(losses["vad_loss"].item()),
            "emotion_loss": float(losses["emotion_loss"].item()),
            "num_samples": int(target.size(0)),
            "num_batches": batches,
        }
    )
    supervised_cccs = []
    for index, name in enumerate(VAD_DIMENSIONS):
        selected = mask[:, index]
        if int(selected.sum().item()) == 0:
            value = None
        else:
            value = float(
                _scalar_ccc(
                    prediction[selected, index],
                    target[selected, index],
                ).item()
            )
            supervised_cccs.append(value)
        result[f"{name}_ccc"] = value
    result["mean_ccc"] = (
        sum(supervised_cccs) / len(supervised_cccs) if supervised_cccs else None
    )
    return result


def save_parallel_checkpoint(
    model,
    output_path,
    class_labels,
    vad_label_counts,
    dominance_status,
    class_names_ja=None,
    lambda_vad=1.0,
    lambda_emo=1.0,
    metadata=None,
    column_config=None,
    vad_normalization=None,
    encoder_info=None,
    training_history=None,
    evaluation_metrics=None,
):
    if dominance_status not in DOMINANCE_STATUSES:
        raise ValueError(f"invalid dominance_status: {dominance_status}")
    counts = _normalize_counts(vad_label_counts)
    labels = list(class_labels)
    if len(labels) != int(model.num_classes):
        raise ValueError("class_labels length must match model.num_classes")
    checkpoint = {
        "model_type": "parallel_emotion_vad",
        "model_state_dict": model.state_dict(),
        "target_dim": 3,
        "input_dim": int(model.input_dim),
        "hidden_dim": int(model.hidden_dim),
        "num_classes": int(model.num_classes),
        "class_labels": labels,
        "class_names_ja": labels if class_names_ja is None else list(class_names_ja),
        "vad_label_counts": counts,
        "supervised_dimensions": [name for name in VAD_DIMENSIONS if counts[name] > 0],
        "dominance_status": dominance_status,
        "lambda_vad": float(lambda_vad),
        "lambda_emo": float(lambda_emo),
        "metadata": {} if metadata is None else metadata,
    }
    # Notebook metadata is additive so checkpoints produced by the existing CLI
    # and older checkpoint readers remain fully compatible.
    if column_config is not None:
        checkpoint["column_config"] = dict(column_config)
    if vad_normalization is not None:
        checkpoint["vad_normalization"] = dict(vad_normalization)
    if encoder_info is not None:
        checkpoint["encoder_info"] = dict(encoder_info)
    if training_history is not None:
        checkpoint["training_history"] = list(training_history)
    if evaluation_metrics is not None:
        checkpoint["evaluation_metrics"] = dict(evaluation_metrics)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def _normalize_counts(counts):
    if isinstance(counts, dict):
        return {name: int(counts.get(name, 0)) for name in VAD_DIMENSIONS}
    values = list(counts)
    if len(values) != 3:
        raise ValueError("vad_label_counts must contain V/A/D counts")
    return {name: int(value) for name, value in zip(VAD_DIMENSIONS, values)}


def _prepare_batch(batch, device, input_key):
    net_input = batch["net_input"]
    if input_key not in net_input:
        raise ValueError(f"batch net_input does not contain '{input_key}'")
    if "vad_target" not in batch or "vad_target_mask" not in batch:
        raise ValueError("batch must contain vad_target and vad_target_mask")
    padding_mask = net_input.get("padding_mask")
    return (
        net_input[input_key].to(device),
        None if padding_mask is None else padding_mask.to(device),
        batch["vad_target"].to(device),
        batch["vad_target_mask"].to(device=device, dtype=torch.bool),
        batch["emotion_target"].to(device),
    )


def _validate_batch(output, vad_target, vad_mask, emotion_target):
    if not isinstance(output, dict) or not {"vad", "logits"}.issubset(output):
        raise ValueError("model output must contain vad and logits")
    if output["vad"].shape != vad_target.shape or vad_target.shape != vad_mask.shape:
        raise ValueError("vad output, target, and mask must have the same shape")
    if vad_target.dim() != 2 or vad_target.size(1) != 3:
        raise ValueError("VAD tensors must have shape [B, 3]")
    if emotion_target.dim() != 1 or emotion_target.size(0) != vad_target.size(0):
        raise ValueError("emotion_target must have shape [B]")


def _scalar_ccc(prediction, target, eps=1e-8):
    prediction_mean = prediction.mean()
    target_mean = target.mean()
    covariance = ((prediction - prediction_mean) * (target - target_mean)).mean()
    denominator = (
        prediction.var(unbiased=False)
        + target.var(unbiased=False)
        + torch.square(prediction_mean - target_mean)
        + eps
    )
    return 2.0 * covariance / denominator
