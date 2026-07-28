"""
IEMOCAPの特徴量（.npy）とラベル（.emo）を読み込み、DataLoaderを生成するデータ管理モジュール。
5-fold交差検証に対応したデータ分割と、パディング付きバッチ生成をサポートする。
"""

import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

def load_dataset(data_path, labels=None, min_length=3, max_length=None):
    """
    .npy（特徴量）と .lengths（フレーム数）、オプションでラベルファイルを読み込む。

    Returns:
        npy_data: 全発話の特徴量を連結した配列 (全フレーム数, D)
        sizes: 各発話のフレーム数リスト
        offsets: npy_data 上での各発話の開始インデックスリスト
        emo_labels: 感情ラベル文字列のリスト（labelsがNoneなら空リスト）
    """
    sizes = []
    offsets = []
    emo_labels = []

    # 全発話の特徴量を一括ロードする（メモリマップ形式で効率的に読み込まれる）
    npy_data = np.load(data_path + ".npy")

    offset = 0
    skipped = 0

    if not os.path.exists(data_path + f".{labels}"):
        labels = None

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[
                1]  # ラベル行のうち、感情カテゴリだけを使う
            # min_length 未満のサンプルはスキップし、max_length 以下のものだけを収集する
            if length >= min_length and (
                max_length is None or length <= max_length
            ):
                sizes.append(length)
                offsets.append(offset)
                if lbl is not None:
                    emo_labels.append(lbl)
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")

    return npy_data, sizes, offsets, emo_labels

class SpeechDataset(Dataset):
    """emotion2vec 特徴量とラベルを保持する PyTorch Dataset クラス。"""

    def __init__(
        self,
        feats,
        sizes,
        offsets,
        labels=None,
        shuffle=True,
        sort_by_length=True,
    ):
        super().__init__()

        self.feats = feats
        self.sizes = sizes      # 各発話サンプルのフレーム数
        self.offsets = offsets  # 連結済み特徴量配列における各発話の開始位置

        self.labels = labels

        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

    def __getitem__(self, index):
        """インデックスに対応する発話特徴量とラベルを返す。"""
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        # .copy() で npy のメモリマップ領域を通常テンソルにコピーする
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        res = {"id": index, "feats": feats}
        if self.labels is not None:
            res["target"] = self.labels[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        """バッチ内の発話を最大長に合わせてゼロパディングし、パディングマスクを生成する。"""
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [s.shape[0] for s in feats]
        labels = torch.tensor([s["target"] for s in samples]) if samples[0]["target"] is not None else None

        target_size = max(sizes)  # バッチ内の最大フレーム数

        # (B, T_max, D) のゼロ行列を用意し、各発話を前詰めでコピーする
        collated_feats = feats[0].new_zeros(
            len(feats), target_size, feats[0].size(-1)
        )

        # パディング部分を True としたマスクを作成する（shape: B x T_max）
        padding_mask = torch.BoolTensor(torch.Size([len(feats), target_size])).fill_(False)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True  # 有効フレーム以降をパディングとしてマークする

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask
            },
            "labels": labels
        }
        return res

    def num_tokens(self, index):
        """fairseq互換の補助メソッド。指定サンプルの系列長を返す。"""
        return self.size(index)

    def size(self, index):
        """指定インデックスの発話フレーム数を返す。"""
        return self.sizes[index]

def load_ssl_features(feature_path, label_dict, max_speech_seq_len=None):
    """特徴量ファイルを読み込み、ラベルを整数IDに変換してまとめた辞書を返す。"""
    data, sizes, offsets, labels = load_dataset(feature_path, labels='emo', min_length=1, max_length=max_speech_seq_len)
    labels = [ label_dict[elem] for elem in labels ]  # 文字列ラベル → 整数ID に変換

    num = len(labels)
    iemocap_data = {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "num": num
    }

    return iemocap_data

def train_valid_test_iemocap_dataloader(
        data,
        batch_size,
        test_start,
        test_end,
        eval_is_test=False,
    ):
    """
    5-fold交差検証用に train / val / test の DataLoader を生成して返す。

    Args:
        test_start, test_end: テストセットとして使うサンプルのインデックス範囲
        eval_is_test: True の場合、テストセットをそのまま検証セットとしても使用する
    """
    feats = data['feats']
    sizes, offsets = data['sizes'], data['offsets']
    labels = data['labels']

    # テストセットのサイズ・オフセット・ラベルを切り出す
    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_labels = labels[test_start:test_end]

    # テストセットの特徴量を npy 配列から抽出し、オフセットを0始まりに再計算する
    test_offset_start = test_offsets[0]
    test_offset_end = test_offsets[-1] + test_sizes[-1]
    test_feats = feats[test_offset_start:test_offset_end, :]
    test_offsets = test_offsets - test_offset_start

    test_dataset = SpeechDataset(
        feats=test_feats,
        sizes=test_sizes,
        offsets=test_offsets,
        labels=test_labels,
    )

    # テストセット以外を連結してトレーニング＋検証データとする
    train_val_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    # オフセットを累積和で再計算する（テストセットを除いた後の連続インデックス）
    train_val_offsets = np.concatenate([np.array([0]), np.cumsum(train_val_sizes)[:-1]], dtype=np.int64)
    train_val_labels = [item for item in labels[:test_start] + labels[test_end:]]
    train_val_feats = np.concatenate([feats[:test_offset_start, :], feats[test_offset_end:, :]], axis=0)

    if eval_is_test:
        # eval_is_test=True のとき: 全非テストデータで学習し、テストセットで検証する
        train_dataset = SpeechDataset(
            feats=train_val_feats,
            sizes=train_val_sizes,
            offsets=train_val_offsets,
            labels=train_val_labels,
        )
        val_dataset = test_dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collator,
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collator,
                                num_workers=4, pin_memory=True, shuffle=False)

    else:
        # eval_is_test=False のとき: 非テストデータを 8:2 でトレーニング/検証に分割する
        train_val_nums = data['num'] - (test_end - test_start)
        train_nums = int(0.8 * train_val_nums)
        val_nums = train_val_nums - train_nums

        train_val_dataset = SpeechDataset(
            feats=train_val_feats,
            sizes=train_val_sizes,
            offsets=train_val_offsets,
            labels=train_val_labels,
        )

        train_dataset, val_dataset = random_split(train_val_dataset, [train_nums, val_nums])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator,
                                num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=train_val_dataset.collator,
                                num_workers=4, pin_memory=True, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collator,
                                num_workers=4, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader
