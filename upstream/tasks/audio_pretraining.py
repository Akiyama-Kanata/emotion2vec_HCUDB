# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
emotion2vec の事前学習タスク定義（fairseq タスク）。
データセットの読み込み・マニフェスト管理・マルチコーパス対応を担う。
fairseq のトレーニングループから呼び出される。
"""

import logging
import os
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, OrderedDict
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import BinarizedAudioDataset, FileAudioDataset, SubsampleDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@dataclass
class AudioMaskingConfig:
    """音声マスク処理の設定をモデル設定から参照するデータクラス。II() でリンクされる。"""
    feature_encoder_spec: str = II("model.modalities.audio.feature_encoder_spec")
    mask_prob: float = II("model.modalities.audio.mask_prob")
    mask_prob_adjust: float = II("model.modalities.audio.mask_prob_adjust")
    mask_length: int = II("model.modalities.audio.mask_length")
    inverse_mask: bool = II("model.modalities.audio.inverse_mask")
    mask_dropout: float = II("model.modalities.audio.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class Emotion2vecPretrainingConfig(FairseqDataclass):
    """emotion2vec 事前学習タスクの設定。データパス・サンプリング・マスク設定を管理する。"""
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    multi_corpus_keys: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated names for loading multi corpus datasets"})
    multi_corpus_sampling_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated string of sampling weights corresponding to the multi_corpus_keys"})
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )

    rebuild_batches: bool = True
    precompute_mask_config: Optional[AudioMaskingConfig] = None

    post_save_script: Optional[str] = None

    subsample: float = 1
    seed: int = II("common.seed")

    sort_indices_mutiple_corpora: Optional[bool] = field(
        default=True,
        metadata={"help": "Sort indices for multiple corpora"}
    )
    batch_sample_multiple_corpora: Optional[bool] = field(
        default=False,
        metadata={"help": "Sample batches from multiple corpora"}
    )


@register_task("emotion2vec_pretraining", dataclass=Emotion2vecPretrainingConfig)
class Emotion2vecPretrainingTask(FairseqTask):
    """fairseq から呼び出される emotion2vec 事前学習タスク本体。"""

    cfg: Emotion2vecPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: Emotion2vecPretrainingConfig, **kwargs):
        """辞書などの追加資源が不要なため、設定を保持したタスクだけを生成する。

        Args:
            cfg (Emotion2vecPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        """
        指定したデータスプリット（train/valid等）のデータセットを読み込んでキャッシュする。
        単一マニフェスト・バイナリ・マルチコーパスの3形式に対応している。
        """
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # 古いチェックポイント由来の Namespace 設定を読み込む場合に、不足属性を補う。
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        # precompute_mask_config が設定されている場合はデータ読み込み時にマスクを事前計算する
        compute_mask = getattr(task_cfg, "precompute_mask_config", None) is not None
        mask_args = {}
        if compute_mask:
            mask_args = task_cfg.precompute_mask_config

        if getattr(task_cfg, "binarized_dataset", False):
            # バイナリ化済みデータセット（大規模データ向け）を読み込む
            self.datasets[split] = BinarizedAudioDataset(
                data_path,
                split=split,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask=compute_mask,
                **mask_args,
            )
        else:
            if task_cfg.multi_corpus_keys is None:
                # 通常の単一TSVマニフェストからデータセットを読み込む
                manifest_path = os.path.join(data_path, "{}.tsv".format(split))

                self.datasets[split] = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                    max_sample_size=self.cfg.max_sample_size,
                    min_sample_size=self.cfg.min_sample_size,
                    pad=task_cfg.labels is not None or task_cfg.enable_padding,
                    normalize=task_cfg.normalize,
                    num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                    text_compression_level=text_compression_level,
                    compute_mask=compute_mask,
                    **mask_args,
                )
            else:
                # マルチコーパス: 複数のTSVをサンプリング重み付きで混合して読み込む
                dataset_map = OrderedDict()
                self.dataset_map = {}
                multi_corpus_keys = [k.strip() for k in task_cfg.multi_corpus_keys.split(",")]
                corpus_idx_map = {k: idx for idx, k in enumerate(multi_corpus_keys)}
                data_keys = [k.split(":") for k in split.split(",")]

                # 各コーパスのサンプリング重みを、multi_corpus_keys と同じ順番で数値化する。
                multi_corpus_sampling_weights = [float(val.strip()) for val in task_cfg.multi_corpus_sampling_weights.split(",")]
                data_weights = []

                for key, file_name in data_keys:

                    k = key.strip()
                    manifest_path = os.path.join(data_path, "{}.tsv".format(file_name.strip()))

                    # 単一コーパスと同じ FileAudioDataset を作り、corpus_key で由来コーパスを識別できるようにする。
                    dataset_map[k] = FileAudioDataset(
                        manifest_path=manifest_path,
                        sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                        max_sample_size=self.cfg.max_sample_size,
                        min_sample_size=self.cfg.min_sample_size,
                        pad=task_cfg.labels is not None or task_cfg.enable_padding,
                        normalize=task_cfg.normalize,
                        num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                        text_compression_level=text_compression_level,
                        compute_mask=compute_mask,
                        corpus_key=corpus_idx_map[k],
                        **mask_args,
                    )

                    data_weights.append(multi_corpus_sampling_weights[corpus_idx_map[k]])

                self.dataset_map[split] = dataset_map
                
                if len(dataset_map) == 1:
                    # 1コーパスだけなら MultiCorpusDataset で包まず、そのまま使う。
                    self.datasets[split] = list(dataset_map.values())[0]
                else:
                    # 複数コーパスの場合は、指定重みに従ってバッチ/サンプルを混合する。
                    self.datasets[split] = MultiCorpusDataset(dataset_map, distribution=data_weights, seed=0, sort_indices=True)

        if getattr(task_cfg, "subsample", 1) < 1:
            # デバッグや高速実験用に、指定割合だけサブサンプリングする。
            self.datasets[split] = SubsampleDataset(
                self.datasets[split],
                task_cfg.subsample,
                shuffle=True,
                seed=task_cfg.seed,
            )

        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    def max_positions(self):
        """モデル側では明示的な最大長を制限せず、fairseq には十分大きい値を返す。"""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        """fairseq の標準手順でモデルを構築し、古い wav2vec 設定があれば引き継ぐ。"""
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model

    def post_save(self, cp_path, num_updates):
        """チェックポイント保存後に、評価用コピー作成と任意の後処理スクリプト実行を行う。"""
        if self.cfg.post_save_script is not None:
            logger.info(f"launching {self.cfg.post_save_script}")
            import os.path as osp
            from fairseq.file_io import PathManager

            eval_cp_path = osp.join(
                osp.dirname(cp_path), f"checkpoint_eval_{num_updates}.pt"
            )

            print(cp_path, eval_cp_path, osp.dirname(cp_path))

            assert PathManager.copy(
                cp_path, eval_cp_path, overwrite=True
            ), f"Failed to copy {cp_path} to {eval_cp_path}"

            import subprocess
            import shlex

            subprocess.call(shlex.split(f"{self.cfg.post_save_script} {eval_cp_path}"))
