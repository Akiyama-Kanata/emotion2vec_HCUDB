"""
VAD空間を中間表現として使う2段階学習アーキテクチャのモデル定義。
AttentionPooling → VADDecoder(FNN) → EmotionClassifier(線形分類器) の構成。
"""

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """スコアアテンションによる可変長フレーム列の発話レベル圧縮。パディング位置を除外して重み付き和を計算する。"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            padding_mask: パディング位置が True の Boolean テンソル (B, T)
        Returns:
            (B, D)
        """
        scores = self.score(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)


class VADDecoder(nn.Module):
    """AttentionPooling + FNN で発話表現を3次元VAD空間に写像する。出力は Tanh で [-1, 1] に正規化。"""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, vad_dim: int = 3):
        super().__init__()
        self.pool = AttentionPooling(input_dim)
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vad_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            padding_mask: (B, T)
        Returns:
            vad: (B, vad_dim)  [Valence, Arousal, Dominance]
        """
        pooled = self.pool(x, padding_mask)  # (B, D)
        return self.fnn(pooled)              # (B, vad_dim)


class EmotionClassifier(nn.Module):
    """VADDecoder の出力を線形分類器でカテゴリ感情に変換する2段階モデル全体。"""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        vad_dim: int = 3,
        num_classes: int = 4,
    ):
        super().__init__()
        self.vad_decoder = VADDecoder(input_dim, hidden_dim, vad_dim)
        self.classifier = nn.Linear(vad_dim, num_classes)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            vad: (B, vad_dim)   VAD推定値（Stage 1損失・可視化用）
            logits: (B, num_classes)  分類ロジット（Stage 2損失用）
        """
        vad = self.vad_decoder(x, padding_mask)
        logits = self.classifier(vad)
        return vad, logits
