from pathlib import Path

import torch
from torch import nn

try:
    from vad_downstream.data import EMOTION_CLASS_LABELS, EMOTION_CLASS_NAMES_JA
    from vad_downstream.training import (
        ccc_loss,
        concordance_correlation_coefficient,
    )
except ModuleNotFoundError:
    from data import EMOTION_CLASS_LABELS, EMOTION_CLASS_NAMES_JA
    from training import ccc_loss, concordance_correlation_coefficient


def compute_vad_emotion_loss(
    output,
    vad_target,
    emotion_target,
    lambda_vad=1.0,
    lambda_emo=1.0,
    emotion_criterion=None,
):
    """Compute CCC + cross-entropy loss for VAD-mediated emotion training."""
    _validate_loss_weights(lambda_vad=lambda_vad, lambda_emo=lambda_emo)
    if not isinstance(output, dict) or "vad" not in output or "logits" not in output:
        raise ValueError("model output must be a dict containing 'vad' and 'logits'")

    if emotion_criterion is None:
        emotion_criterion = nn.CrossEntropyLoss()

    vad_loss = ccc_loss(output["vad"], vad_target)
    emotion_loss = emotion_criterion(output["logits"], emotion_target)
    total_loss = float(lambda_vad) * vad_loss + float(lambda_emo) * emotion_loss
    return {
        "loss": total_loss,
        "vad_loss": vad_loss,
        "emotion_loss": emotion_loss,
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
    emotion_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_vad_loss = 0.0
    total_emotion_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        model_input, padding_mask, vad_target, emotion_target = _prepare_batch(
            batch,
            device=device,
            input_key=input_key,
        )

        optimizer.zero_grad()
        output = model(model_input, padding_mask=padding_mask, return_vad=True)
        losses = compute_vad_emotion_loss(
            output,
            vad_target,
            emotion_target,
            lambda_vad=lambda_vad,
            lambda_emo=lambda_emo,
            emotion_criterion=emotion_criterion,
        )
        losses["loss"].backward()
        optimizer.step()

        total_loss += float(losses["loss"].detach().cpu().item())
        total_vad_loss += float(losses["vad_loss"].detach().cpu().item())
        total_emotion_loss += float(losses["emotion_loss"].detach().cpu().item())
        num_batches += 1

    if num_batches == 0:
        raise ValueError("data_loader must yield at least one batch")

    return {
        "loss": total_loss / num_batches,
        "vad_loss": total_vad_loss / num_batches,
        "emotion_loss": total_emotion_loss / num_batches,
        "num_batches": int(num_batches),
    }


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
    vad_predictions = []
    vad_targets = []
    logits = []
    emotion_targets = []
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            model_input, padding_mask, vad_target, emotion_target = _prepare_batch(
                batch,
                device=device,
                input_key=input_key,
            )
            output = model(model_input, padding_mask=padding_mask, return_vad=True)

            vad_predictions.append(output["vad"].detach().cpu())
            vad_targets.append(vad_target.detach().cpu())
            logits.append(output["logits"].detach().cpu())
            emotion_targets.append(emotion_target.detach().cpu())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("data_loader must yield at least one batch")

    vad_prediction = torch.cat(vad_predictions, dim=0)
    vad_target = torch.cat(vad_targets, dim=0)
    emotion_logits = torch.cat(logits, dim=0)
    emotion_target = torch.cat(emotion_targets, dim=0)

    losses = compute_vad_emotion_loss(
        {"vad": vad_prediction, "logits": emotion_logits},
        vad_target,
        emotion_target,
        lambda_vad=lambda_vad,
        lambda_emo=lambda_emo,
    )
    ccc = concordance_correlation_coefficient(vad_prediction, vad_target)
    class_metrics = classification_metrics(
        emotion_logits,
        emotion_target,
        class_labels=class_labels,
    )

    metrics = {
        "loss": float(losses["loss"].detach().cpu().item()),
        "vad_loss": float(losses["vad_loss"].detach().cpu().item()),
        "emotion_loss": float(losses["emotion_loss"].detach().cpu().item()),
        "valence_ccc": float(ccc[0].detach().cpu().item()),
        "arousal_ccc": float(ccc[1].detach().cpu().item()),
        "mean_ccc": float(ccc.mean().detach().cpu().item()),
        "wa": class_metrics["wa"],
        "ua": class_metrics["ua"],
        "weighted_f1": class_metrics["weighted_f1"],
        "macro_f1": class_metrics["macro_f1"],
        "confusion_matrix": class_metrics["confusion_matrix"],
        "per_class_recall": class_metrics["per_class_recall"],
        "per_class_f1": class_metrics["per_class_f1"],
        "class_support": class_metrics["class_support"],
        "class_labels": class_metrics["class_labels"],
        "num_samples": int(vad_target.size(0)),
        "num_batches": int(num_batches),
    }
    if vad_target.size(1) == 3:
        metrics["dominance_ccc"] = float(ccc[2].detach().cpu().item())
    return metrics


def classification_metrics(logits, target, num_classes=None, class_labels=None):
    """Return WA, UA, weighted F1, and a target-row/prediction-column matrix."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, C], got {logits.shape}")
    if target.dim() != 1:
        raise ValueError(f"target must be 1D [B], got {target.shape}")
    if logits.size(0) != target.size(0):
        raise ValueError(
            f"logits and target batch sizes must match, got "
            f"{logits.size(0)} and {target.size(0)}"
        )
    if logits.size(0) == 0:
        raise ValueError("metrics require at least one sample")

    inferred_num_classes = int(logits.size(1))
    if num_classes is None:
        num_classes = inferred_num_classes
    num_classes = int(num_classes)
    if num_classes != inferred_num_classes:
        raise ValueError(
            f"num_classes ({num_classes}) does not match logits dim "
            f"({inferred_num_classes})"
        )

    labels = _resolve_class_labels(num_classes, class_labels)
    target = target.to(dtype=torch.long, device="cpu")
    prediction = logits.argmax(dim=1).to(dtype=torch.long, device="cpu")
    if torch.any(target < 0) or torch.any(target >= num_classes):
        raise ValueError("target contains an out-of-range class index")

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for truth, pred in zip(target.tolist(), prediction.tolist()):
        confusion[int(truth), int(pred)] += 1

    total = int(target.numel())
    correct = int(torch.diag(confusion).sum().item())
    support = confusion.sum(dim=1).to(dtype=torch.float32)
    predicted = confusion.sum(dim=0).to(dtype=torch.float32)
    true_positive = torch.diag(confusion).to(dtype=torch.float32)

    recalls = []
    f1_values = []
    per_class_recall = {}
    per_class_f1 = {}
    class_support = {}
    for index, label in enumerate(labels):
        class_support[label] = int(support[index].item())
        if support[index].item() > 0:
            recall = float((true_positive[index] / support[index]).item())
            recalls.append(recall)
            per_class_recall[label] = recall
        else:
            recall = 0.0
            per_class_recall[label] = None

        if predicted[index].item() > 0:
            precision = float((true_positive[index] / predicted[index]).item())
        else:
            precision = 0.0

        if precision + recall > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_values.append(f1)
        per_class_f1[label] = f1 if support[index].item() > 0 else None

    weighted_f1 = 0.0
    for index, f1 in enumerate(f1_values):
        weighted_f1 += f1 * float(support[index].item()) / float(total)

    return {
        "wa": correct / float(total),
        "ua": sum(recalls) / float(len(recalls)) if recalls else 0.0,
        "weighted_f1": weighted_f1,
        "macro_f1": sum(f1_values) / float(num_classes),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "class_support": class_support,
        "class_labels": labels,
    }


def save_vad_emotion_checkpoint(
    model,
    output_path,
    target_dim=None,
    class_labels=None,
    class_names_ja=None,
    lambda_vad=1.0,
    lambda_emo=1.0,
    metadata=None,
):
    _validate_loss_weights(lambda_vad=lambda_vad, lambda_emo=lambda_emo)
    target_dim = _module_int_attr(model, "target_dim", target_dim)
    input_dim = _module_int_attr(model, "input_dim", 768)
    hidden_dim = _module_int_attr(model, "hidden_dim", 256)
    num_classes = _module_int_attr(model, "num_classes", None)
    if target_dim not in (2, 3):
        raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")

    if class_labels is None:
        class_labels = EMOTION_CLASS_LABELS
    class_labels = list(class_labels)
    if num_classes is None:
        num_classes = len(class_labels)
    if len(class_labels) != int(num_classes):
        raise ValueError(
            f"class_labels length ({len(class_labels)}) does not match "
            f"num_classes ({int(num_classes)})"
        )
    if class_names_ja is None:
        class_names_ja = EMOTION_CLASS_NAMES_JA
    class_names_ja = list(class_names_ja)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "target_dim": int(target_dim),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "num_classes": int(num_classes),
        "class_labels": class_labels,
        "class_names_ja": class_names_ja,
        "lambda_vad": float(lambda_vad),
        "lambda_emo": float(lambda_emo),
        "metadata": {} if metadata is None else metadata,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def copy_cpu_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _prepare_batch(batch, device, input_key="feats"):
    net_input = batch["net_input"]
    if input_key not in net_input:
        raise ValueError(f"batch net_input does not contain '{input_key}'")

    model_input = net_input[input_key].to(device)
    padding_mask = net_input.get("padding_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)

    vad_target = batch.get("vad_target", batch.get("target"))
    if vad_target is None:
        raise ValueError("batch must contain 'vad_target' or 'target'")
    if "emotion_target" not in batch:
        raise ValueError("batch must contain 'emotion_target'")

    return (
        model_input,
        padding_mask,
        vad_target.to(device),
        batch["emotion_target"].to(device),
    )


def _validate_loss_weights(lambda_vad, lambda_emo):
    if lambda_vad < 0.0:
        raise ValueError("lambda_vad must be non-negative")
    if lambda_emo < 0.0:
        raise ValueError("lambda_emo must be non-negative")
    if lambda_vad == 0.0 and lambda_emo == 0.0:
        raise ValueError("at least one loss weight must be positive")


def _resolve_class_labels(num_classes, class_labels=None):
    if class_labels is None:
        if num_classes == len(EMOTION_CLASS_LABELS):
            return list(EMOTION_CLASS_LABELS)
        return [f"class_{index}" for index in range(num_classes)]

    labels = list(class_labels)
    if len(labels) != num_classes:
        raise ValueError(
            f"class_labels length ({len(labels)}) does not match "
            f"num_classes ({num_classes})"
        )
    return labels


def _module_int_attr(module, name, fallback):
    if hasattr(module, name):
        return int(getattr(module, name))
    if hasattr(module, "head") and hasattr(module.head, name):
        return int(getattr(module.head, name))
    if fallback is None:
        return None
    return int(fallback)
