"""
IEMOCAPの特徴量（.npy）、感情ラベル（.emo）、VA連続値アノテーションを読み込むデータ管理モジュール。
既存の iemocap_downstream/data.py をベースに va_labels フィールドを追加。
"""

import contextlib
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


def load_dataset(data_path, labels=None, min_length=3, max_length=None):
    """
    .npy / .lengths / .{labels} を読み込む（iemocap_downstream と同一ロジック）。

    Returns:
        npy_data, sizes, offsets, emo_labels
    """
    sizes = []
    offsets = []
    emo_labels = []

    npy_data = np.load(data_path + ".npy")

    offset = 0
    skipped = 0

    if labels is not None and not os.path.exists(data_path + f".{labels}"):
        labels = None

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl = None if labels is None else next(lbl_f).rstrip().split()[1]
            if length >= min_length and (max_length is None or length <= max_length):
                sizes.append(length)
                offsets.append(offset)
                if lbl is not None:
                    emo_labels.append(lbl)
            else:
                skipped += 1
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")
    return npy_data, sizes, offsets, emo_labels


def load_iemocap_with_va(feature_path: str, label_dict: dict, va_path: str) -> dict:
    """
    カテゴリラベル + Valence/Arousal 連続値を同時に読み込む。

    va_path は1行1サンプルのテキストファイルで、各行が
        <utterance_id> <valence> <arousal>
    の形式であることを想定する。

    Returns:
        feats, sizes, offsets, labels(int), va_labels(float array), num
    """
    data, sizes, offsets, emo_labels = load_dataset(
        feature_path, labels="emo", min_length=1
    )
    labels = [label_dict[e] for e in emo_labels]

    # VA値を読み込む（行順が特徴量と一致している前提）
    va_values = []
    with open(va_path, "r") as f:
        for line in f:
            parts = line.rstrip().split()
            valence = float(parts[1])
            arousal = float(parts[2])
            va_values.append([valence, arousal])

    va_labels = np.array(va_values, dtype=np.float32)  # (N, 2)

    assert len(labels) == len(va_labels), (
        f"ラベル数不一致: categories={len(labels)}, va={len(va_labels)}"
    )

    return {
        "feats": data,
        "sizes": sizes,
        "offsets": offsets,
        "labels": labels,
        "va_labels": va_labels,
        "num": len(labels),
    }


class SpeechDatasetVAD(Dataset):
    """emotion2vec 特徴量・カテゴリラベル・VA連続値ラベルを保持する Dataset。"""

    def __init__(self, feats, sizes, offsets, labels=None, va_labels=None):
        super().__init__()
        self.feats = feats
        self.sizes = sizes
        self.offsets = offsets
        self.labels = labels
        self.va_labels = va_labels  # numpy (N, 2) or None

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = offset + self.sizes[index]
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        res = {"id": index, "feats": feats}
        if self.labels is not None:
            res["target"] = self.labels[index]
        if self.va_labels is not None:
            res["va_target"] = torch.from_numpy(self.va_labels[index].copy())
        return res

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [s["feats"] for s in samples]
        sizes = [f.shape[0] for f in feats]
        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(len(feats), target_size, feats[0].size(-1))
        padding_mask = torch.zeros(len(feats), target_size, dtype=torch.bool)

        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"feats": collated_feats, "padding_mask": padding_mask},
        }

        if samples[0].get("target") is not None:
            res["labels"] = torch.tensor([s["target"] for s in samples])

        if samples[0].get("va_target") is not None:
            res["va_labels"] = torch.stack([s["va_target"] for s in samples])  # (B, 2)

        return res


def build_dataloaders(data: dict, batch_size: int, test_start: int, test_end: int, eval_is_test: bool = False):
    """
    5-fold 交差検証用に train / val / test DataLoader を生成する。

    テストセット範囲は [test_start, test_end) インデックスで指定。
    """
    feats = data["feats"]
    sizes, offsets = data["sizes"], data["offsets"]
    labels = data["labels"]
    va_labels = data.get("va_labels")

    # --- テストセット抽出 ---
    test_sizes = sizes[test_start:test_end]
    test_offsets = offsets[test_start:test_end]
    test_labels = labels[test_start:test_end]
    test_va = va_labels[test_start:test_end] if va_labels is not None else None

    t_off_start = test_offsets[0]
    t_off_end = test_offsets[-1] + test_sizes[-1]
    test_feats = feats[t_off_start:t_off_end, :]
    test_offsets = test_offsets - t_off_start

    test_ds = SpeechDatasetVAD(test_feats, test_sizes, test_offsets, test_labels, test_va)

    # --- 訓練＋検証セット構築 ---
    tv_sizes = np.concatenate([sizes[:test_start], sizes[test_end:]])
    tv_offsets = np.concatenate([np.array([0]), np.cumsum(tv_sizes)[:-1]]).astype(np.int64)
    tv_labels = labels[:test_start] + labels[test_end:]
    tv_feats = np.concatenate([feats[:t_off_start, :], feats[t_off_end:, :]], axis=0)
    tv_va = (
        np.concatenate([va_labels[:test_start], va_labels[test_end:]], axis=0)
        if va_labels is not None
        else None
    )

    if eval_is_test:
        train_ds = SpeechDatasetVAD(tv_feats, tv_sizes, tv_offsets, tv_labels, tv_va)
        val_ds = test_ds
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collator,
                                  num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=val_ds.collator,
                                num_workers=4, pin_memory=True, shuffle=False)
    else:
        n_tv = len(tv_labels)
        n_train = int(0.8 * n_tv)
        n_val = n_tv - n_train

        tv_ds = SpeechDatasetVAD(tv_feats, tv_sizes, tv_offsets, tv_labels, tv_va)
        train_ds, val_ds = random_split(tv_ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=tv_ds.collator,
                                  num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=tv_ds.collator,
                                num_workers=4, pin_memory=True, shuffle=False)

    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=test_ds.collator,
                             num_workers=4, pin_memory=True, shuffle=False)

    return train_loader, val_loader, test_loader
