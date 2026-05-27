"""
IEMOCAPの特徴量（.npy）、感情ラベル（.emo）、VA連続値アノテーションを読み込むデータ管理モジュール。
既存の iemocap_downstream/data.py をベースに va_labels フィールドを追加。
"""

import contextlib
import csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

WAGNER_VAD_COLUMNS = ("arousal", "dominance", "valence")
_SPLIT_ALIASES = {
    "train": "train",
    "tr": "train",
    "dev": "val",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
    "eval": "test",
}


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

    utt_ids = []

    with open(data_path + ".lengths", "r") as len_f, open(
        data_path + f".{labels}", "r"
    ) if labels is not None else contextlib.ExitStack() as lbl_f:
        for line in len_f:
            length = int(line.rstrip())
            lbl_line = None if labels is None else next(lbl_f).rstrip()
            if length >= min_length and (max_length is None or length <= max_length):
                sizes.append(length)
                offsets.append(offset)
                if lbl_line is not None:
                    parts = lbl_line.split()
                    utt_ids.append(parts[0])
                    emo_labels.append(parts[1])
            else:
                skipped += 1
            offset += length

    sizes = np.asarray(sizes)
    offsets = np.asarray(offsets)

    logger.info(f"loaded {len(offsets)}, skipped {skipped} samples")
    return npy_data, sizes, offsets, emo_labels, utt_ids


def load_iemocap_with_va(feature_path: str, label_dict: dict, va_path: str) -> dict:
    """
    カテゴリラベル + Valence/Arousal 連続値を同時に読み込む。

    va_path は1行1サンプルのテキストファイルで、各行が
        <utterance_id> <valence> <arousal>
    の形式であることを想定する。

    Returns:
        feats, sizes, offsets, labels(int), va_labels(float array), num
    """
    data, sizes, offsets, emo_labels, utt_ids = load_dataset(
        feature_path, labels="emo", min_length=1
    )
    labels = [label_dict[e] for e in emo_labels]

    # VA値を発話IDをキーとして辞書に読み込む
    va_dict: dict[str, list[float]] = {}
    with open(va_path, "r") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) >= 3:
                va_dict[parts[0]] = [float(parts[1]), float(parts[2])]

    # 特徴量ファイルの発話順にVA値を取得
    missing = [uid for uid in utt_ids if uid not in va_dict]
    if missing:
        raise ValueError(
            f"va_labels.txt に存在しない発話ID: {missing[:5]} ... (計{len(missing)}件)"
        )

    va_values = [va_dict[uid] for uid in utt_ids]
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
                                  num_workers=0, pin_memory=False, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=val_ds.collator,
                                num_workers=0, pin_memory=False, shuffle=False)
    else:
        n_tv = len(tv_labels)
        n_train = int(0.8 * n_tv)
        n_val = n_tv - n_train

        tv_ds = SpeechDatasetVAD(tv_feats, tv_sizes, tv_offsets, tv_labels, tv_va)
        train_ds, val_ds = random_split(tv_ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=tv_ds.collator,
                                  num_workers=0, pin_memory=False, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=tv_ds.collator,
                                num_workers=0, pin_memory=False, shuffle=False)

    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=test_ds.collator,
                             num_workers=0, pin_memory=False, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Wagner互換 VAD 回帰用: CSV + WAV + cached emotion2vec features
# ---------------------------------------------------------------------------


def _lookup_field(fieldnames: Sequence[str], name: str) -> Optional[str]:
    fields = {field.strip().lower(): field for field in fieldnames}
    return fields.get(name)


def _parse_vad_value(raw_value: str, column: str, line_number: int) -> float:
    value = str(raw_value).strip()
    if value == "":
        return float("nan")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{column} must be numeric at CSV line {line_number}: {value!r}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(
            f"{column} must be in the Wagner-compatible 0..1 range at CSV line {line_number}: {parsed}"
        )
    return parsed


def load_vad_csv(csv_path: str, audio_dir: Optional[str] = None) -> List[Dict[str, object]]:
    """
    CSV から Wagner互換 VAD 回帰用レコードを読み込む。

    必須列:
        file_path
    ラベル列:
        arousal / dominance / valence のうち少なくとも1列
    任意列:
        split, session
    """
    csv_file = Path(csv_path)
    base_dir = Path(audio_dir) if audio_dir else csv_file.parent

    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        file_field = _lookup_field(fieldnames, "file_path")
        if file_field is None:
            raise ValueError("CSV must contain a file_path column.")

        vad_fields = {
            column: _lookup_field(fieldnames, column)
            for column in WAGNER_VAD_COLUMNS
        }
        if not any(vad_fields.values()):
            raise ValueError("CSV must contain at least one of: arousal, dominance, valence.")

        split_field = _lookup_field(fieldnames, "split")
        session_field = _lookup_field(fieldnames, "session")

        records: List[Dict[str, object]] = []
        for row_index, row in enumerate(reader):
            line_number = row_index + 2
            raw_file = str(row.get(file_field, "")).strip()
            if raw_file == "":
                raise ValueError(f"file_path is empty at CSV line {line_number}.")

            audio_path = Path(raw_file)
            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path

            record: Dict[str, object] = {
                "index": row_index,
                "file_path": str(audio_path),
            }
            has_label = False
            for column, field in vad_fields.items():
                value = float("nan") if field is None else _parse_vad_value(row.get(field, ""), column, line_number)
                record[column] = value
                has_label = has_label or np.isfinite(value)

            if not has_label:
                raise ValueError(
                    f"CSV line {line_number} has no usable VAD label. "
                    "At least one of arousal/dominance/valence is required."
                )

            if split_field is not None:
                record["split"] = str(row.get(split_field, "")).strip()
            if session_field is not None:
                record["session"] = str(row.get(session_field, "")).strip()

            records.append(record)

    if not records:
        raise ValueError("CSV contains no data rows.")
    return records


def feature_cache_path(audio_path: str, cache_dir: str, index: int) -> Path:
    """音声パスと行番号から衝突しにくい feature cache path を作る。"""
    source = str(audio_path)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:12]
    stem = Path(source).stem or "audio"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)[:48]
    return Path(cache_dir) / f"{index:06d}_{safe_stem}_{digest}.npy"


def attach_cache_paths(records: Sequence[Dict[str, object]], cache_dir: str) -> List[Dict[str, object]]:
    """各レコードに cache_path を付与したコピーを返す。"""
    prepared: List[Dict[str, object]] = []
    for offset, record in enumerate(records):
        copied = dict(record)
        index = int(copied.get("index", offset))
        copied["cache_path"] = str(feature_cache_path(str(copied["file_path"]), cache_dir, index))
        prepared.append(copied)
    return prepared


def ensure_feature_cache(
    records: Sequence[Dict[str, object]],
    extractor: Callable[[str], object],
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, object]]:
    """
    emotion2vec 特徴を .npy キャッシュへ保存する。

    extractor は file_path を受け取り、(T, D) または (D,) の numpy/torch/list を返す callable。
    返り値は cache_path を持つレコードのコピー。
    """
    prepared = attach_cache_paths(records, cache_dir) if cache_dir is not None else [dict(r) for r in records]

    for record in prepared:
        if "cache_path" not in record:
            raise ValueError("cache_path is missing. Pass cache_dir or call attach_cache_paths first.")
        cache_path = Path(str(record["cache_path"]))
        if cache_path.exists() and not force:
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feats = extractor(str(record["file_path"]))
        if torch.is_tensor(feats):
            feats = feats.detach().cpu().numpy()
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        elif feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats[0]
        if feats.ndim != 2:
            raise ValueError(
                f"extractor must return a 2-D feature array for {record['file_path']}, got shape {feats.shape}"
            )
        np.save(cache_path, feats)

    return prepared


def _normalise_split_name(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(f"unknown split value {value!r}; expected train/val/test.")
    return _SPLIT_ALIASES[key]


def _random_train_val_test_split(
    records: Sequence[Dict[str, object]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    if len(records) < 3:
        n_test = 0
        n_val = 0
    else:
        n_test = max(1, int(round(len(records) * test_ratio))) if test_ratio > 0 else 0
        n_val = max(1, int(round(len(records) * val_ratio))) if val_ratio > 0 and len(records) >= 4 else 0
        if n_test + n_val >= len(records):
            n_test = 1
            n_val = 0

    test_idx = set(indices[:n_test].tolist())
    val_idx = set(indices[n_test:n_test + n_val].tolist())
    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []
    test: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        if idx in test_idx:
            test.append(dict(record))
        elif idx in val_idx:
            val.append(dict(record))
        else:
            train.append(dict(record))
    return {"train": train, "val": val, "test": test}


def split_vad_records(
    records: Sequence[Dict[str, object]],
    mode: str = "auto",
    test_session: Optional[str] = None,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[Dict[str, object]]]:
    """
    split 優先順位に従って train/val/test を作る。

    auto の場合:
        1. split 列が全行にあるならそれを使う
        2. session 列があるなら leave-one-session-out
        3. seed 固定 random 80/10/10
    """
    if not records:
        raise ValueError("records is empty.")

    selected_mode = (mode or "auto").strip().lower()
    has_split_column = any("split" in record for record in records)
    has_session = any(str(record.get("session", "")).strip() for record in records)

    if selected_mode == "auto":
        if has_split_column:
            selected_mode = "split"
        elif has_session:
            selected_mode = "session"
        else:
            selected_mode = "random"

    if selected_mode == "split":
        splits = {"train": [], "val": [], "test": []}
        for record in records:
            split_name = _normalise_split_name(record.get("split", ""))
            splits[split_name].append(dict(record))
        if not splits["train"]:
            raise ValueError("explicit split must include at least one train row.")
        return splits

    if selected_mode == "session":
        sessions = sorted({str(record.get("session", "")).strip() for record in records if str(record.get("session", "")).strip()})
        if not sessions:
            raise ValueError("session split requested, but no session values were found.")
        held_out = str(test_session).strip() if test_session is not None else sessions[0]
        if held_out not in sessions:
            raise ValueError(f"test_session {held_out!r} was not found. Available sessions: {sessions}")

        train_pool = [dict(record) for record in records if str(record.get("session", "")).strip() != held_out]
        test = [dict(record) for record in records if str(record.get("session", "")).strip() == held_out]
        val_split = _random_train_val_test_split(train_pool, seed=seed, val_ratio=val_ratio, test_ratio=0.0)
        return {"train": val_split["train"], "val": val_split["val"], "test": test}

    if selected_mode == "random":
        return _random_train_val_test_split(records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    raise ValueError("mode must be one of: auto, split, session, random.")


def leave_one_session_out_splits(
    records: Sequence[Dict[str, object]],
    seed: int = 42,
    val_ratio: float = 0.1,
) -> List[Tuple[str, Dict[str, List[Dict[str, object]]]]]:
    """全 session を1回ずつ test にする split 一覧を返す。"""
    sessions = sorted({str(record.get("session", "")).strip() for record in records if str(record.get("session", "")).strip()})
    return [
        (
            session,
            split_vad_records(
                records,
                mode="session",
                test_session=session,
                seed=seed,
                val_ratio=val_ratio,
                test_ratio=0.0,
            ),
        )
        for session in sessions
    ]


class VADFeatureDataset(Dataset):
    """CSVレコードと cache済み emotion2vec 特徴から VAD 回帰バッチを作る Dataset。"""

    def __init__(self, records: Sequence[Dict[str, object]]):
        super().__init__()
        self.records = [dict(record) for record in records]

    def __getitem__(self, index):
        record = self.records[index]
        feats = np.load(str(record["cache_path"])).astype(np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.ndim != 2:
            raise ValueError(f"cached features must be 2-D, got {feats.shape}: {record['cache_path']}")

        labels = np.asarray([record.get(column, np.nan) for column in WAGNER_VAD_COLUMNS], dtype=np.float32)
        mask = np.isfinite(labels)
        labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)

        return {
            "id": int(record.get("index", index)),
            "file_path": str(record["file_path"]),
            "feats": torch.from_numpy(feats).float(),
            "vad_target": torch.from_numpy(labels).float(),
            "vad_mask": torch.from_numpy(mask),
        }

    def __len__(self):
        return len(self.records)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [sample["feats"] for sample in samples]
        sizes = [feat.shape[0] for feat in feats]
        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(len(feats), target_size, feats[0].size(-1))
        padding_mask = torch.zeros(len(feats), target_size, dtype=torch.bool)
        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        return {
            "id": torch.LongTensor([sample["id"] for sample in samples]),
            "file_path": [sample["file_path"] for sample in samples],
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask,
            },
            "vad_labels": torch.stack([sample["vad_target"] for sample in samples]),
            "vad_mask": torch.stack([sample["vad_mask"] for sample in samples]).bool(),
        }


def build_vad_dataloader(
    records: Sequence[Dict[str, object]],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = VADFeatureDataset(records)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
