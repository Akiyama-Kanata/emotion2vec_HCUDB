"""
VAD回帰タスク用の損失関数モジュール。
Concordance Correlation Coefficient (CCC) に基づいた損失を定義する。
"""

import torch
from typing import Optional


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CCC損失 (1 - CCC) を計算する。

    Args:
        pred:   (B,) の予測値
        target: (B,) の正解値
    Returns:
        スカラー損失値
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()

    ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
    return 1.0 - ccc


def stage1_loss(vad_pred: torch.Tensor, va_target: torch.Tensor) -> torch.Tensor:
    """
    Stage 1用損失: Valence と Arousal の CCC損失の和（Dominanceラベルなし）。

    Args:
        vad_pred:  (B, 3) — VADDecoder の出力 [V, A, D]
        va_target: (B, 2) — VAラベル [V, A]
    Returns:
        スカラー損失値
    """
    loss_v = ccc_loss(vad_pred[:, 0], va_target[:, 0])
    loss_a = ccc_loss(vad_pred[:, 1], va_target[:, 1])
    return loss_v + loss_a


def vad_ccc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wagner互換 VAD 回帰用の CCC loss。

    Args:
        pred:        (B, 3) — [arousal, dominance, valence] の 0..1 予測
        target:      (B, 3) — 同じ順序の正解。欠損値は NaN でもよい
        target_mask: (B, 3) — 利用可能なラベル位置が True。None の場合は finite(target)
    Returns:
        利用可能な次元だけで平均した CCC loss。全ラベル欠損時は 0 loss を返す
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
