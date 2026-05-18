"""
VAD回帰タスク用の損失関数モジュール。
Concordance Correlation Coefficient (CCC) に基づいた損失を定義する。
"""

import torch


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
