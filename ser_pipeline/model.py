"""Dataset-independent decoder compatible with the legacy BaseModel state dict."""

from __future__ import annotations

import torch
from torch import nn


class BaseModel(nn.Module):
    """Two linear layers over masked mean-pooled frame features."""

    def __init__(self, input_dim=768, output_dim=4, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_probability = float(dropout)
        self.pre_net = nn.Linear(self.input_dim, self.hidden_dim)
        self.post_net = nn.Linear(self.hidden_dim, self.output_dim)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_probability)

    def forward(self, x, padding_mask=None):
        if x.ndim != 3:
            raise ValueError(f"features must be 3D [batch, frames, dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"feature dim must be {self.input_dim}, got {x.shape[-1]}")
        if x.shape[1] <= 0:
            raise ValueError("features must contain at least one frame")
        if padding_mask is None:
            padding_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        if padding_mask.shape != x.shape[:2]:
            raise ValueError("padding_mask shape must equal [batch, frames]")
        padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
        valid = ~padding_mask
        counts = valid.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("every sample must contain at least one non-padding frame")
        hidden = self.dropout(self.activate(self.pre_net(x)))
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pooled = hidden.sum(dim=1) / counts.to(hidden.dtype)
        return self.post_net(pooled)


__all__ = ["BaseModel"]
