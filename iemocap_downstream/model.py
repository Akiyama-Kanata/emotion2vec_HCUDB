"""
IEMOCAP感情認識タスク用の2層線形分類器を定義するモジュール。
emotion2vec 特徴量（768次元）を受け取り、感情クラス（ang/hap/neu/sad）を予測する。
"""

import torch
from torch import nn


class BaseModel(nn.Module):
    """2層全結合ネットワークによる感情分類器。パディングマスク付き平均プーリングで発話レベルの表現を得る。"""

    def __init__(self, input_dim=768, output_dim=4, hidden_dim=256, dropout=0.0):
        super().__init__()
        # 第1層: emotion2vec 特徴量を調整可能な中間表現へ変換
        self.pre_net = nn.Linear(input_dim, hidden_dim)
        # 第2層: 中間表現から感情クラス数の出力へ変換（ロジット）
        self.post_net = nn.Linear(hidden_dim, output_dim)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: 入力特徴量テンソル (B, T, input_dim)
            padding_mask: パディング位置を True とするマスク (B, T)
        Returns:
            感情クラスのロジット (B, output_dim)
        """
        x = self.dropout(self.activate(self.pre_net(x)))

        # パディング位置をゼロで埋めてから、有効フレームのみを平均する（マスク付き平均プーリング）
        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)  # Compute average

        x = self.post_net(x)  # (B, output_dim)
        return x
