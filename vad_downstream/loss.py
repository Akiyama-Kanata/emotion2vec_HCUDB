"""
Loss functions for Wagner-compatible VAD regression.

Predictions and labels use the public order:
    arousal, dominance, valence
"""

from typing import Optional

import torch


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Concordance Correlation Coefficient loss, defined as 1 - CCC.

    Args:
        pred: Prediction tensor with shape (B,).
        target: Target tensor with shape (B,).
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()

    ccc = (2 * covariance) / (
        pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    )
    return 1.0 - ccc


def vad_ccc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    CCC loss averaged over available VAD dimensions.

    Args:
        pred: Tensor with shape (B, 3), ordered as arousal, dominance, valence.
        target: Tensor with shape (B, 3), same order. Missing labels may be NaN.
        target_mask: Boolean tensor with shape (B, 3), where True means usable label.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape: {pred.shape} != {target.shape}")
    if pred.ndim != 2:
        raise ValueError(f"expected (B, D) tensors, got shape {tuple(pred.shape)}")

    if target_mask is None:
        target_mask = torch.isfinite(target)
    else:
        target_mask = target_mask.to(dtype=torch.bool, device=target.device)
        target_mask = target_mask & torch.isfinite(target)

    loss = pred.sum() * 0.0
    n_dims = 0
    for dim in range(pred.size(1)):
        valid = target_mask[:, dim]
        if valid.any():
            loss = loss + ccc_loss(pred[valid, dim], target[valid, dim])
            n_dims += 1

    if n_dims == 0:
        return loss
    return loss / n_dims
