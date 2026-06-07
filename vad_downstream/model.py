"""
VAD regression head for cached emotion2vec frame features.

The public output order is always:
    arousal, dominance, valence
"""

from typing import Dict, Optional

import torch
from torch import nn


VAD_OUTPUT_NAMES = ("arousal", "dominance", "valence")


class MaskedMeanPooling(nn.Module):
    """Pool variable-length frame features while excluding padded frames."""

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Feature tensor with shape (B, T, D).
            padding_mask: Boolean tensor with shape (B, T), where True means padding.
        Returns:
            Utterance-level tensor with shape (B, D).
        """
        if padding_mask is None:
            return x.mean(dim=1)

        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        summed = (x * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return summed / denom


class Emotion2VecVADRegressor(nn.Module):
    """
    Wagner/audeering-compatible VAD regressor for frozen emotion2vec features.

    The model expects frame-level or utterance-level emotion2vec features and returns
    0..1 scores in VAD_OUTPUT_NAMES order: arousal, dominance, valence.
    """

    output_names = VAD_OUTPUT_NAMES

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = MaskedMeanPooling()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.output_names)),
            nn.Sigmoid(),
        )

    def forward(
        self, feats: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            feats: emotion2vec features with shape (B, T, D).
            padding_mask: Boolean tensor with shape (B, T), where True means padding.
        Returns:
            Tensor with shape (B, 3), ordered as arousal, dominance, valence.
        """
        pooled = self.pool(feats, padding_mask)
        return self.regressor(pooled)


def vad_tensor_to_dict(pred: torch.Tensor) -> Dict[str, float]:
    """Convert one VAD prediction tensor to a named dict."""
    if pred.ndim == 2:
        if pred.size(0) != 1:
            raise ValueError("vad_tensor_to_dict expects a single prediction, got a batch.")
        pred = pred.squeeze(0)
    if pred.ndim != 1 or pred.numel() != len(VAD_OUTPUT_NAMES):
        raise ValueError(f"expected shape (3,), got {tuple(pred.shape)}")
    values = pred.detach().cpu().float().tolist()
    return {name: float(value) for name, value in zip(VAD_OUTPUT_NAMES, values)}
