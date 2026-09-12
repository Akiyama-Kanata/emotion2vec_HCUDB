"""Dataset-independent decoder compatible with the legacy BaseModel state dict."""

from __future__ import annotations

import torch
from torch import nn
import copy


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


class OfficialHeadModel(nn.Module):
    """Independent official nine-row projection over valid-frame means."""

    def __init__(self, head, *, condition="C"):
        super().__init__()
        if condition not in {"C", "D"}:
            raise ValueError("official condition must be C or D")
        self.condition = condition
        self.input_dim, self.output_dim = 1024, 9
        self.label_spec = head.label_spec
        self.provenance = copy.deepcopy(head.provenance)
        if not isinstance(head.proj, nn.Linear) or head.proj.bias is None or head.proj.weight.dtype != torch.float32 or head.proj.bias.dtype != torch.float32:
            raise ValueError("official head must be an FP32 nn.Linear with bias")
        if "head_sha256" in self.provenance:
            from .official import state_sha256
            if state_sha256({"proj.weight": head.proj.weight, "proj.bias": head.proj.bias}) != self.provenance["head_sha256"]:
                raise ValueError("official initial head hash mismatch")
        self.proj = copy.deepcopy(head.proj).float()
        if self.proj.weight.shape != (9, 1024) or self.proj.bias.shape != (9,):
            raise ValueError("official proj must be Linear(1024, 9) with bias")
        self.requires_grad_(condition == "D")
        self.eval()

    def train(self, mode=True):
        if mode and self.condition == "C":
            raise ValueError("condition C is a frozen baseline and cannot train")
        return super().train(mode)

    def forward(self, features, padding_mask=None):
        if features.ndim != 3 or features.shape[0] == 0 or features.shape[1] == 0 or features.shape[2] != 1024:
            raise ValueError("official features must be nonempty [B, T, 1024]")
        if features.dtype != torch.float32:
            raise ValueError("official features must be FP32")
        if padding_mask is None:
            padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
        if padding_mask.shape != features.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool [B, T]")
        padding_mask = padding_mask.to(features.device)
        counts = (~padding_mask).sum(1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("every sample requires non-padding frames")
        clean = features.masked_fill(padding_mask[..., None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("non-finite valid features")
        return self.proj(clean.sum(1) / counts.to(features.dtype))


__all__ = ["BaseModel", "OfficialHeadModel"]
