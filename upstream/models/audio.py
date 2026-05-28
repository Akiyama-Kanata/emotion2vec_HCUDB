# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
音声モダリティ専用エンコーダの定義。
CNN特徴抽出器（ConvFeatureExtractionModel）と畳み込み位置エンコーダで構成され、
ModalitySpecificEncoder（base.py）を継承する。
"""

from functools import partial
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import (
    LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.tasks import FairseqTask
from .base import D2vModalityConfig, ModalitySpecificEncoder, get_alibi_bias
from .modules import Modality, BlockEncoder, Decoder1d

@dataclass
class D2vAudioConfig(D2vModalityConfig):
    """音声モダリティエンコーダのハイパーパラメータ設定。"""
    type: Modality = Modality.AUDIO
    extractor_mode: str = "layer_norm"
    feature_encoder_spec: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_pos_width: int = field(
        default=95,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_depth: int = field(
        default=5,
        metadata={"help": "depth of positional encoder network"},
    )
    conv_pos_pre_ln: bool = False  # True の場合、畳み込み位置エンコーダの前に LayerNorm を挟む


class AudioEncoder(ModalitySpecificEncoder):
    """
    音声モダリティ専用エンコーダ。
    CNN特徴抽出 → 線形射影 → 畳み込み位置エンコーダ → prenet Transformer の順で処理する。
    """

    modality_cfg: D2vAudioConfig

    def __init__(
        self,
        modality_cfg: D2vAudioConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
    ):
        # CNN層の仕様をeval()で評価して[(dim, kernel, stride), ...]のリストに変換する
        self.feature_enc_layers = eval(modality_cfg.feature_encoder_spec)
        feature_embed_dim = self.feature_enc_layers[-1][0]  # CNN最終層の出力次元

        # 波形から局所特徴量を抽出する多層CNN（wav2vec2と同一アーキテクチャ）
        local_encoder = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=modality_cfg.extractor_mode,
            conv_bias=False,
        )

        # CNN特徴量の次元を embed_dim に揃える線形射影
        project_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        )

        # 畳み込み相対位置エンコーダ（depth 層の Conv1d + LayerNorm + GELU のスタック）
        num_pos_layers = modality_cfg.conv_pos_depth
        k = max(3, modality_cfg.conv_pos_width // num_pos_layers)  # 各層のカーネル幅

        positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=modality_cfg.conv_pos_groups,  # グループ畳み込みで計算量を削減
                    ),
                    SamePad(k),   # 入力と同じ系列長を維持するパディング除去
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        if modality_cfg.conv_pos_pre_ln:
            positional_encoder = nn.Sequential(LayerNorm(embed_dim), positional_encoder)

        # prenet: Transformer ブロックの Drop Path率を線形補間で割り当てる
        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        # デコーダ（事前学習時のみ使用。skip_ema=True の場合は None になる）
        decoder = (
            Decoder1d(modality_cfg.decoder, embed_dim)
            if modality_cfg.decoder is not None
            else None
        )

        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=None,
            relative_positional_encoder=positional_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def convert_padding_mask(self, x, padding_mask):
        """
        入力波形のパディングマスクを、CNN後のフレーム系列に対応したマスクに変換する。
        各CNNレイヤーのカーネル幅とストライドを適用して出力長を計算する。
        """
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """CNN特徴抽出器を通過した後の系列長を、畳み込みの出力長公式で計算する。"""

            def _conv_out_length(input_length, kernel_size, stride):
                """1つのConv1d層に対する出力長 floor((L-k)/s+1) を返す。"""
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                # 各CNN層を順番に適用した場合の有効フレーム数へ更新していく。
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # 入力波形上の有効長を、CNN後のフレーム数に変換する。
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # 各サンプルの最後の有効フレーム位置に印を付ける。
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                # flip → cumsum → flip により、最後の有効位置より後だけをパディング True にする。
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    x.shape[:2], dtype=torch.bool, device=x.device
                )

        return padding_mask

    def reset_parameters(self):
        """線形射影とデコーダのパラメータを再初期化する。"""
        super().reset_parameters()
        for mod in self.project_features.children():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()
