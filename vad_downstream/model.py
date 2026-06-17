import contextlib

import torch
from torch import nn


class Emotion2vecVADModel(nn.Module):
    """Whole VA/VAD model: audio waveform -> emotion2vec frames -> regression."""

    def __init__(
        self,
        encoder,
        target_dim,
        input_dim=768,
        hidden_dim=256,
        freeze_encoder=True,
    ):
        super().__init__()
        if target_dim not in (2, 3):
            raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")

        self.encoder = encoder
        self.target_dim = target_dim
        self.input_dim = input_dim
        self.freeze_encoder = freeze_encoder
        self.head = VADRegressionHead(
            target_dim=target_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, source, padding_mask=None):
        features, feature_padding_mask = self.extract_frame_features(
            source, padding_mask=padding_mask
        )
        return self.head(features, feature_padding_mask)

    def extract_frame_features(self, source, padding_mask=None):
        context = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with context:
            encoder_out = self._call_encoder(source, padding_mask)

        return _parse_encoder_output(encoder_out)

    def _call_encoder(self, source, padding_mask):
        if hasattr(self.encoder, "extract_features"):
            try:
                return self.encoder.extract_features(
                    source,
                    padding_mask=padding_mask,
                    mask=False,
                    remove_extra_tokens=True,
                )
            except TypeError:
                return self.encoder.extract_features(
                    source,
                    padding_mask=padding_mask,
                )

        try:
            return self.encoder(source, padding_mask=padding_mask)
        except TypeError:
            return self.encoder(source)


class VADRegressionHead(nn.Module):
    """VA/VAD regression head for padded frame-level emotion2vec features."""

    def __init__(self, target_dim, input_dim=768, hidden_dim=256):
        super().__init__()
        if target_dim not in (2, 3):
            raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")

        self.target_dim = target_dim
        self.input_dim = input_dim
        self.pre_net = nn.Linear(input_dim, hidden_dim)
        self.post_net = nn.Linear(hidden_dim, target_dim)
        self.activate = nn.ReLU()

    def forward(self, features, padding_mask=None):
        if features.dim() != 3:
            raise ValueError(
                f"emotion2vec features must be 3D [B, T, C], got {features.shape}"
            )
        if features.size(-1) != self.input_dim:
            raise ValueError(
                f"emotion2vec feature dim must be {self.input_dim}, got "
                f"{features.size(-1)}"
            )

        if padding_mask is None:
            feature_padding_mask = torch.zeros(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        else:
            feature_padding_mask = padding_mask.to(
                device=features.device, dtype=torch.bool
            )

        x = masked_mean_pooling(features, feature_padding_mask)
        x = self.activate(self.pre_net(x))
        return self.post_net(x)


def masked_mean_pooling(features, padding_mask):
    """Mean-pool frame features while ignoring padded frames."""
    if padding_mask.shape != features.shape[:2]:
        raise ValueError(
            f"padding_mask shape must be {features.shape[:2]}, got "
            f"{padding_mask.shape}"
        )

    valid_mask = ~padding_mask
    valid_counts = valid_mask.sum(dim=1)
    if torch.any(valid_counts == 0):
        raise ValueError("cannot pool an utterance with no valid frames")

    masked_features = features * valid_mask.unsqueeze(-1).type_as(features)
    return masked_features.sum(dim=1) / valid_counts.unsqueeze(-1).type_as(features)


def _parse_encoder_output(encoder_out):
    if isinstance(encoder_out, dict):
        if "x" not in encoder_out:
            raise ValueError("encoder output dict must contain an 'x' tensor")
        return encoder_out["x"], encoder_out.get("padding_mask")

    if isinstance(encoder_out, tuple):
        if len(encoder_out) == 2:
            return encoder_out
        if len(encoder_out) == 1:
            return encoder_out[0], None
        raise ValueError("encoder output tuple must be (features, padding_mask)")

    if torch.is_tensor(encoder_out):
        return encoder_out, None

    raise ValueError("encoder output must be a dict, tuple, or tensor")
