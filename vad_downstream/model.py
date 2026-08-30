"""連続感情回帰、VAD経由分類、並列感情・VAD推定に使うモデルを定義する。"""

import contextlib

import torch
from torch import nn


VAD_OUTPUT_NAMES = ("valence", "arousal", "dominance")


class MaskedMeanPooling(nn.Module):
    """Pool frame features while excluding positions marked as padding."""

    def forward(self, features, padding_mask=None):
        if features.dim() != 3:
            raise ValueError(
                f"emotion2vec features must be 3D [B, T, C], got {features.shape}"
            )
        if padding_mask is None:
            return features.mean(dim=1)

        padding_mask = padding_mask.to(device=features.device, dtype=torch.bool)
        if padding_mask.shape != features.shape[:2]:
            raise ValueError(
                "padding_mask shape must match the first two feature dimensions: "
                f"{padding_mask.shape} != {features.shape[:2]}"
            )
        if padding_mask.all(dim=1).any():
            raise ValueError("cannot pool a fully padded feature sequence")

        valid = (~padding_mask).unsqueeze(-1).to(dtype=features.dtype)
        return (features * valid).sum(dim=1) / valid.sum(dim=1)


class Emotion2VecVADRegressor(nn.Module):
    """Compatibility regressor for precomputed emotion2vec frame features."""

    output_names = VAD_OUTPUT_NAMES

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
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

    def forward(self, feats, padding_mask=None):
        return self.regressor(self.pool(feats, padding_mask))


def vad_tensor_to_dict(pred):
    """Convert one VAD prediction into a name-to-score mapping."""
    if pred.ndim == 2:
        if pred.size(0) != 1:
            raise ValueError("vad_tensor_to_dict expects a single prediction")
        pred = pred.squeeze(0)
    if pred.ndim != 1 or pred.numel() != len(VAD_OUTPUT_NAMES):
        raise ValueError(f"expected shape (3,), got {tuple(pred.shape)}")
    return {
        name: float(value)
        for name, value in zip(
            VAD_OUTPUT_NAMES, pred.detach().cpu().float().tolist()
        )
    }


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
        self.hidden_dim = hidden_dim
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


class VADClassificationHead(nn.Module):
    """Logistic-regression classifier over predicted VA/VAD values."""

    def __init__(self, target_dim, num_classes):
        super().__init__()
        if target_dim not in (2, 3):
            raise ValueError(f"target_dim must be 2 or 3, got {target_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")

        self.target_dim = target_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(target_dim, num_classes)

    def forward(self, vad):
        if vad.dim() != 2:
            raise ValueError(f"VAD input must be 2D [B, D], got {vad.shape}")
        if vad.size(1) != self.target_dim:
            raise ValueError(
                f"VAD input dim must be {self.target_dim}, got {vad.size(1)}"
            )

        return self.linear(vad)


class VADMediatedEmotionClassifier(nn.Module):
    """Emotion classifier whose class logits depend only on predicted VA/VAD."""

    def __init__(
        self,
        target_dim=3,
        num_classes=4,
        input_dim=768,
        hidden_dim=256,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vad_head = VADRegressionHead(
            target_dim=target_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        self.classifier = VADClassificationHead(
            target_dim=target_dim,
            num_classes=num_classes,
        )

    def forward(self, features, padding_mask=None, return_vad=False):
        vad = self.vad_head(features, padding_mask=padding_mask)
        logits = self.classifier(vad)
        if return_vad:
            return {
                "vad": vad,
                "logits": logits,
            }
        return logits


class IndependentRegressionHead(nn.Module):
    """One scalar affect head kept independent from every other task head."""

    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pre_net = nn.Linear(input_dim, hidden_dim)
        self.activate = nn.ReLU()
        self.post_net = nn.Linear(hidden_dim, 1)

    def forward(self, pooled_features):
        return self.post_net(self.activate(self.pre_net(pooled_features)))


class ParallelEmotionVADClassifier(nn.Module):
    """Independent emotion and V/A/D heads over one masked pooled feature."""

    target_dim = 3

    def __init__(self, num_classes=4, input_dim=768, hidden_dim=256):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emotion_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.valence_head = IndependentRegressionHead(input_dim, hidden_dim)
        self.arousal_head = IndependentRegressionHead(input_dim, hidden_dim)
        self.dominance_head = IndependentRegressionHead(input_dim, hidden_dim)

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
            padding_mask = torch.zeros(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        else:
            padding_mask = padding_mask.to(device=features.device, dtype=torch.bool)

        pooled = masked_mean_pooling(features, padding_mask)
        vad = torch.cat(
            [
                self.valence_head(pooled),
                self.arousal_head(pooled),
                self.dominance_head(pooled),
            ],
            dim=1,
        )
        return {"logits": self.emotion_head(pooled), "vad": vad}

    def task_parameters(self, include_dominance=True):
        """Return optimizer parameters, optionally excluding all D parameters."""
        modules = [self.emotion_head, self.valence_head, self.arousal_head]
        if include_dominance:
            modules.append(self.dominance_head)
        return [parameter for module in modules for parameter in module.parameters()]


class Emotion2vecParallelEmotionVADClassifier(Emotion2vecVADModel):
    """Frozen waveform encoder followed by independent parallel task heads."""

    def __init__(
        self,
        encoder,
        num_classes=4,
        input_dim=768,
        hidden_dim=256,
        freeze_encoder=True,
    ):
        super().__init__(
            encoder=encoder,
            target_dim=3,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
        )
        self.num_classes = num_classes
        self.head = ParallelEmotionVADClassifier(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, source, padding_mask=None):
        features, feature_padding_mask = self.extract_frame_features(
            source, padding_mask=padding_mask
        )
        return self.head(features, padding_mask=feature_padding_mask)


class Emotion2vecVADMediatedClassifier(Emotion2vecVADModel):
    """Whole classifier: audio waveform -> emotion2vec frames -> VAD -> logits."""

    def __init__(
        self,
        encoder,
        target_dim=3,
        num_classes=4,
        input_dim=768,
        hidden_dim=256,
        freeze_encoder=True,
    ):
        super().__init__(
            encoder=encoder,
            target_dim=target_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
        )
        self.num_classes = num_classes
        self.head = VADMediatedEmotionClassifier(
            target_dim=target_dim,
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, source, padding_mask=None, return_vad=False):
        features, feature_padding_mask = self.extract_frame_features(
            source,
            padding_mask=padding_mask,
        )
        return self.head(
            features,
            padding_mask=feature_padding_mask,
            return_vad=return_vad,
        )


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
