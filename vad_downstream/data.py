import csv
import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

FEATURE_DIM = 768
EMOTION_CLASS_LABELS = ["hap", "sad", "ang", "dis"]
EMOTION_CLASS_NAMES_JA = ["喜び", "悲しみ", "怒り", "嫌悪"]
WAGNER_VAD_COLUMNS = ("valence", "arousal", "dominance")
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


def load_vad_dataset(prefix, min_length=1, max_length=None):
    """Load frame-level emotion2vec features with utterance-level VA/VAD labels."""
    feats, lengths = _load_features_and_lengths(prefix)
    utt_ids, targets, target_dim = _load_vad_labels(prefix + ".vad")
    if len(utt_ids) != len(lengths):
        raise ValueError(
            f"number of VAD rows ({len(utt_ids)}) does not match lengths "
            f"({len(lengths)})"
        )

    sizes_all, offsets_all, keep = _build_size_index(
        lengths,
        min_length=min_length,
        max_length=max_length,
    )

    sizes = sizes_all[keep]
    offsets = offsets_all[keep]
    targets = targets[keep]
    utt_ids = [utt_ids[i] for i in keep]

    logger.info("loaded %d VAD samples, skipped %d", len(sizes), len(lengths) - len(sizes))

    return {
        "feats": feats,
        "sizes": sizes,
        "offsets": offsets,
        "targets": targets,
        "utt_ids": utt_ids,
        "target_dim": target_dim,
        "num": len(sizes),
    }


def load_vad_emotion_dataset(
    prefix,
    min_length=1,
    max_length=None,
    class_labels=None,
    masked_vad=True,
):
    """Load frame-level features with aligned VAD and categorical emotion labels."""
    class_labels = _normalize_class_labels(class_labels)
    feats, lengths = _load_features_and_lengths(prefix)

    if masked_vad:
        vad_utt_ids, vad_targets, vad_target_masks = _load_masked_vad_labels(
            prefix + ".vad"
        )
        target_dim = 3
    else:
        vad_utt_ids, vad_targets, target_dim = _load_vad_labels(prefix + ".vad")
        vad_target_masks = np.ones_like(vad_targets, dtype=np.bool_)
    if len(vad_utt_ids) != len(lengths):
        raise ValueError(
            f"number of VAD rows ({len(vad_utt_ids)}) does not match lengths "
            f"({len(lengths)})"
        )

    emo_utt_ids, emotion_targets, emotion_labels = _load_emotion_labels(
        prefix + ".emo",
        class_labels=class_labels,
    )
    if len(emo_utt_ids) != len(lengths):
        raise ValueError(
            f"number of emotion rows ({len(emo_utt_ids)}) does not match lengths "
            f"({len(lengths)})"
        )

    for i, (vad_utt_id, emo_utt_id) in enumerate(
        zip(vad_utt_ids, emo_utt_ids),
        start=1,
    ):
        if vad_utt_id != emo_utt_id:
            raise ValueError(
                f"utterance_id mismatch at row {i}: VAD has {vad_utt_id!r}, "
                f"emotion has {emo_utt_id!r}"
            )

    sizes_all, offsets_all, keep = _build_size_index(
        lengths,
        min_length=min_length,
        max_length=max_length,
    )

    sizes = sizes_all[keep]
    offsets = offsets_all[keep]
    vad_targets = vad_targets[keep]
    vad_target_masks = vad_target_masks[keep]
    emotion_targets = emotion_targets[keep]
    utt_ids = [vad_utt_ids[i] for i in keep]
    emotion_labels = [emotion_labels[i] for i in keep]

    logger.info(
        "loaded %d VAD-emotion samples, skipped %d",
        len(sizes),
        len(lengths) - len(sizes),
    )

    return {
        "feats": feats,
        "sizes": sizes,
        "offsets": offsets,
        "targets": vad_targets,
        "vad_targets": vad_targets,
        "vad_target_masks": vad_target_masks,
        "vad_label_counts": vad_target_masks.sum(axis=0).astype(np.int64),
        "emotion_targets": emotion_targets,
        "emotion_labels": emotion_labels,
        "utt_ids": utt_ids,
        "target_dim": target_dim,
        "class_labels": class_labels,
        "class_names_ja": list(EMOTION_CLASS_NAMES_JA),
        "num_classes": len(class_labels),
        "num": len(sizes),
    }


class VADSpeechDataset(Dataset):
    def __init__(self, feats, sizes, offsets, targets, utt_ids=None):
        super().__init__()
        self.feats = feats
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.utt_ids = (
            list(utt_ids)
            if utt_ids is not None
            else [str(i) for i in range(len(self.sizes))]
        )

        num_samples = len(self.sizes)
        if len(self.offsets) != num_samples:
            raise ValueError("sizes and offsets must have the same length")
        if len(self.targets) != num_samples:
            raise ValueError("targets and sizes must have the same length")
        if len(self.utt_ids) != num_samples:
            raise ValueError("utt_ids and sizes must have the same length")

    def __getitem__(self, index):
        offset = int(self.offsets[index])
        end = offset + int(self.sizes[index])
        feats = torch.from_numpy(self.feats[offset:end, :].copy()).float()

        return {
            "id": index,
            "utt_id": self.utt_ids[index],
            "feats": feats,
            "target": torch.from_numpy(self.targets[index].copy()).float(),
        }

    def __len__(self):
        return len(self.sizes)

    def collator(self, samples):
        if len(samples) == 0:
            return {}

        feats = [sample["feats"] for sample in samples]
        sizes = [feat.shape[0] for feat in feats]
        target_size = max(sizes)

        collated_feats = feats[0].new_zeros(
            len(feats), target_size, feats[0].size(-1)
        )
        padding_mask = torch.zeros(len(feats), target_size, dtype=torch.bool)

        for i, (feat, size) in enumerate(zip(feats, sizes)):
            collated_feats[i, :size] = feat
            padding_mask[i, size:] = True

        return {
            "id": torch.LongTensor([sample["id"] for sample in samples]),
            "utt_id": [sample["utt_id"] for sample in samples],
            "net_input": {
                "feats": collated_feats,
                "padding_mask": padding_mask,
            },
            "target": torch.stack([sample["target"] for sample in samples]),
        }

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]


class VADEmotionSpeechDataset(VADSpeechDataset):
    def __init__(
        self,
        feats,
        sizes,
        offsets,
        vad_targets,
        emotion_targets,
        utt_ids=None,
        emotion_labels=None,
        class_labels=None,
        vad_target_masks=None,
    ):
        super().__init__(feats, sizes, offsets, vad_targets, utt_ids=utt_ids)
        if vad_target_masks is None:
            vad_target_masks = np.ones_like(self.targets, dtype=np.bool_)
        self.vad_target_masks = np.asarray(vad_target_masks, dtype=np.bool_)
        if self.vad_target_masks.shape != self.targets.shape:
            raise ValueError("vad_target_masks must have the same shape as vad_targets")
        self.class_labels = _normalize_class_labels(class_labels)
        self.emotion_targets = np.asarray(emotion_targets, dtype=np.int64)
        if len(self.emotion_targets) != len(self.sizes):
            raise ValueError("emotion_targets and sizes must have the same length")
        if np.any(self.emotion_targets < 0) or np.any(
            self.emotion_targets >= len(self.class_labels)
        ):
            raise ValueError("emotion_targets contains an out-of-range class index")

        if emotion_labels is None:
            self.emotion_labels = [
                self.class_labels[int(target)] for target in self.emotion_targets
            ]
        else:
            self.emotion_labels = list(emotion_labels)
            if len(self.emotion_labels) != len(self.sizes):
                raise ValueError("emotion_labels and sizes must have the same length")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["vad_target"] = sample["target"]
        sample["vad_target_mask"] = torch.from_numpy(
            self.vad_target_masks[index].copy()
        ).bool()
        sample["emotion_target"] = torch.tensor(
            int(self.emotion_targets[index]),
            dtype=torch.long,
        )
        sample["emotion_label"] = self.emotion_labels[index]
        return sample

    def collator(self, samples):
        batch = super().collator(samples)
        if len(samples) == 0:
            return batch

        batch["vad_target"] = batch["target"]
        batch["vad_target_mask"] = torch.stack(
            [sample["vad_target_mask"] for sample in samples]
        )
        batch["emotion_target"] = torch.LongTensor(
            [int(sample["emotion_target"]) for sample in samples]
        )
        batch["emotion_label"] = [sample["emotion_label"] for sample in samples]
        return batch


def _load_features_and_lengths(prefix):
    feats = np.load(prefix + ".npy")
    _validate_features(feats, prefix + ".npy")

    lengths = _load_lengths(prefix + ".lengths")
    total_length = int(np.sum(lengths, dtype=np.int64))
    if total_length != feats.shape[0]:
        raise ValueError(
            f"sum of lengths ({total_length}) does not match feature frames "
            f"({feats.shape[0]})"
        )
    return feats, lengths


def _build_size_index(lengths, min_length=1, max_length=None):
    sizes_all = np.asarray(lengths, dtype=np.int64)
    offsets_all = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(sizes_all, dtype=np.int64)[:-1]]
    )

    keep = []
    for i, size in enumerate(sizes_all):
        if size >= min_length and (max_length is None or size <= max_length):
            keep.append(i)

    return sizes_all, offsets_all, keep


def _validate_features(feats, path):
    if feats.ndim != 2:
        raise ValueError(f"{path} must be a 2D array, got shape {feats.shape}")
    if feats.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{path} must have feature dimension {FEATURE_DIM}, got {feats.shape[1]}"
        )


def _load_lengths(path):
    lengths = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"empty length at {path}:{line_no}")
            try:
                length = int(text)
            except ValueError as exc:
                raise ValueError(f"invalid length at {path}:{line_no}: {text}") from exc
            if length < 0:
                raise ValueError(f"negative length at {path}:{line_no}: {length}")
            lengths.append(length)

    return lengths


def _load_vad_labels(path):
    utt_ids = []
    targets = []
    target_dim = None

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.rstrip("\r\n")
            if not text:
                raise ValueError(f"empty VAD row at {path}:{line_no}")

            parts = text.split("\t")
            if len(parts) not in (3, 4):
                raise ValueError(
                    f"VAD row at {path}:{line_no} must have 3 or 4 tab-separated "
                    f"columns, got {len(parts)}"
                )

            row_target_dim = len(parts) - 1
            if target_dim is None:
                target_dim = row_target_dim
            elif row_target_dim != target_dim:
                raise ValueError(
                    f"VAD target dimension changes at {path}:{line_no}: "
                    f"expected {target_dim}, got {row_target_dim}"
                )

            utt_id = parts[0]
            if not utt_id:
                raise ValueError(f"empty utterance_id at {path}:{line_no}")

            try:
                values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"invalid VAD value at {path}:{line_no}") from exc

            if np.any(values < -1.0) or np.any(values > 1.0):
                raise ValueError(
                    f"VAD values at {path}:{line_no} must be in [-1.0, 1.0]"
                )

            utt_ids.append(utt_id)
            targets.append(values)

    if target_dim is None:
        raise ValueError(f"{path} has no VAD rows")

    return utt_ids, np.stack(targets).astype(np.float32), target_dim


def _load_masked_vad_labels(path):
    """Load mixed VA/VAD rows into fixed [N, 3] targets and boolean masks."""
    utt_ids = []
    targets = []
    masks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.rstrip("\r\n")
            if not text:
                raise ValueError(f"empty VAD row at {path}:{line_no}")
            parts = text.split("\t")
            if len(parts) not in (3, 4):
                raise ValueError(
                    f"VAD row at {path}:{line_no} must have 3 or 4 tab-separated "
                    f"columns, got {len(parts)}"
                )
            if not parts[0]:
                raise ValueError(f"empty utterance_id at {path}:{line_no}")
            try:
                values = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"invalid VAD value at {path}:{line_no}") from exc
            if np.any(np.asarray(values) < -1.0) or np.any(np.asarray(values) > 1.0):
                raise ValueError(
                    f"VAD values at {path}:{line_no} must be in [-1.0, 1.0]"
                )
            has_dominance = len(values) == 3
            utt_ids.append(parts[0])
            targets.append(values if has_dominance else values + [0.0])
            masks.append([True, True, has_dominance])
    if not utt_ids:
        raise ValueError(f"{path} has no VAD rows")
    return (
        utt_ids,
        np.asarray(targets, dtype=np.float32),
        np.asarray(masks, dtype=np.bool_),
    )


def _load_emotion_labels(path, class_labels=None):
    class_labels = _normalize_class_labels(class_labels)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    utt_ids = []
    targets = []
    labels = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.rstrip("\r\n")
            if not text:
                raise ValueError(f"empty emotion row at {path}:{line_no}")

            parts = text.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"emotion row at {path}:{line_no} must have 2 tab-separated "
                    f"columns, got {len(parts)}"
                )

            utt_id, label = parts
            if not utt_id:
                raise ValueError(f"empty utterance_id at {path}:{line_no}")
            if label not in label_to_index:
                expected = ", ".join(class_labels)
                raise ValueError(
                    f"unknown emotion label at {path}:{line_no}: {label!r}; "
                    f"expected one of: {expected}"
                )

            utt_ids.append(utt_id)
            targets.append(label_to_index[label])
            labels.append(label)

    if not utt_ids:
        raise ValueError(f"{path} has no emotion rows")

    return utt_ids, np.asarray(targets, dtype=np.int64), labels


def _normalize_class_labels(class_labels=None):
    if class_labels is None:
        class_labels = EMOTION_CLASS_LABELS
    class_labels = list(class_labels)
    if len(class_labels) < 2:
        raise ValueError("class_labels must contain at least two classes")
    if len(set(class_labels)) != len(class_labels):
        raise ValueError("class_labels must not contain duplicates")
    if any(not label for label in class_labels):
        raise ValueError("class_labels must not contain empty labels")
    return class_labels


# Compatibility API for the cached-feature VAD regression CLI.  These helpers
# predate the frame-level dataset functions above, but train_vad.py and existing
# research workflows still use them.
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
        raise ValueError(
            f"{column} must be numeric at CSV line {line_number}: {value!r}"
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(
            f"{column} must be in the 0..1 range at CSV line "
            f"{line_number}: {parsed}"
        )
    return parsed


def load_vad_csv(
    csv_path: str, audio_dir: Optional[str] = None
) -> List[Dict[str, object]]:
    """Load cached-feature VAD records from a CSV annotation file."""
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
            raise ValueError(
                "CSV must contain at least one of: valence, arousal, dominance."
            )

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
                value = (
                    float("nan")
                    if field is None
                    else _parse_vad_value(row.get(field, ""), column, line_number)
                )
                record[column] = value
                has_label = has_label or np.isfinite(value)

            if not has_label:
                raise ValueError(
                    f"CSV line {line_number} has no usable VAD label."
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
    """Build a stable cache filename from the audio path and CSV row index."""
    source = str(audio_path)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:12]
    stem = Path(source).stem or "audio"
    safe_stem = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in stem
    )[:48]
    return Path(cache_dir) / f"{index:06d}_{safe_stem}_{digest}.npy"


def attach_cache_paths(
    records: Sequence[Dict[str, object]], cache_dir: str
) -> List[Dict[str, object]]:
    """Return copies of records with deterministic ``cache_path`` values.

    Cache hashes created from absolute paths differ between Windows and WSL.
    Reuse a uniquely matching row/stem cache when one already exists so a cache
    generated on either side remains portable.
    """
    prepared: List[Dict[str, object]] = []
    for offset, record in enumerate(records):
        copied = dict(record)
        index = int(copied.get("index", offset))
        audio_path = str(copied["file_path"])
        cache_path = feature_cache_path(audio_path, cache_dir, index)
        if not cache_path.exists():
            stem = Path(audio_path).stem or "audio"
            safe_stem = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in stem
            )[:48]
            matches = list(
                Path(cache_dir).glob(f"{index:06d}_{safe_stem}_*.npy")
            )
            if len(matches) == 1:
                cache_path = matches[0]
        copied["cache_path"] = str(cache_path)
        prepared.append(copied)
    return prepared


def ensure_feature_cache(
    records: Sequence[Dict[str, object]],
    extractor: Callable[[str], object],
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, object]]:
    """Create missing ``.npy`` feature caches with the supplied extractor."""
    prepared = (
        attach_cache_paths(records, cache_dir)
        if cache_dir is not None
        else [dict(record) for record in records]
    )

    for record in prepared:
        if "cache_path" not in record:
            raise ValueError(
                "cache_path is missing. Pass cache_dir or call attach_cache_paths first."
            )
        cache_path = Path(str(record["cache_path"]))
        if cache_path.exists() and not force:
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features = extractor(str(record["file_path"]))
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        elif features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        if features.ndim != 2:
            raise ValueError(
                "extractor must return a 2-D feature array for "
                f"{record['file_path']}, got shape {features.shape}"
            )
        np.save(cache_path, features)

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
        n_test = (
            max(1, int(round(len(records) * test_ratio))) if test_ratio > 0 else 0
        )
        n_val = (
            max(1, int(round(len(records) * val_ratio)))
            if val_ratio > 0 and len(records) >= 4
            else 0
        )
        if n_test + n_val >= len(records):
            n_test = 1
            n_val = 0

    test_indices = set(indices[:n_test].tolist())
    val_indices = set(indices[n_test : n_test + n_val].tolist())
    splits: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for index, record in enumerate(records):
        if index in test_indices:
            split_name = "test"
        elif index in val_indices:
            split_name = "val"
        else:
            split_name = "train"
        splits[split_name].append(dict(record))
    return splits


def split_vad_records(
    records: Sequence[Dict[str, object]],
    mode: str = "auto",
    test_session: Optional[str] = None,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[Dict[str, object]]]:
    """Split records by explicit labels, held-out session, or seeded random split."""
    if not records:
        raise ValueError("records is empty.")

    selected_mode = (mode or "auto").strip().lower()
    has_split_column = any("split" in record for record in records)
    has_session = any(
        str(record.get("session", "")).strip() for record in records
    )
    if selected_mode == "auto":
        if has_split_column:
            selected_mode = "split"
        elif has_session:
            selected_mode = "session"
        else:
            selected_mode = "random"

    if selected_mode == "split":
        splits: Dict[str, List[Dict[str, object]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for record in records:
            split_name = _normalise_split_name(record.get("split", ""))
            splits[split_name].append(dict(record))
        if not splits["train"]:
            raise ValueError("explicit split must include at least one train row.")
        return splits

    if selected_mode == "session":
        sessions = sorted(
            {
                str(record.get("session", "")).strip()
                for record in records
                if str(record.get("session", "")).strip()
            }
        )
        if not sessions:
            raise ValueError("session split requested, but no session values were found.")
        held_out = str(test_session).strip() if test_session is not None else sessions[0]
        if held_out not in sessions:
            raise ValueError(
                f"test_session {held_out!r} was not found. Available sessions: {sessions}"
            )
        train_pool = [
            dict(record)
            for record in records
            if str(record.get("session", "")).strip() != held_out
        ]
        test = [
            dict(record)
            for record in records
            if str(record.get("session", "")).strip() == held_out
        ]
        train_val = _random_train_val_test_split(
            train_pool, seed=seed, val_ratio=val_ratio, test_ratio=0.0
        )
        return {"train": train_val["train"], "val": train_val["val"], "test": test}

    if selected_mode == "random":
        return _random_train_val_test_split(
            records, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio
        )
    raise ValueError("mode must be one of: auto, split, session, random.")


def leave_one_session_out_splits(
    records: Sequence[Dict[str, object]],
    seed: int = 42,
    val_ratio: float = 0.1,
) -> List[Tuple[str, Dict[str, List[Dict[str, object]]]]]:
    sessions = sorted(
        {
            str(record.get("session", "")).strip()
            for record in records
            if str(record.get("session", "")).strip()
        }
    )
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
    """Dataset backed by cached frame features and utterance-level VAD labels."""

    def __init__(self, records: Sequence[Dict[str, object]]):
        super().__init__()
        self.records = [dict(record) for record in records]

    def __getitem__(self, index):
        record = self.records[index]
        features = np.load(str(record["cache_path"])).astype(np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2:
            raise ValueError(
                f"cached features must be 2-D, got {features.shape}: "
                f"{record['cache_path']}"
            )

        labels = np.asarray(
            [record.get(column, np.nan) for column in WAGNER_VAD_COLUMNS],
            dtype=np.float32,
        )
        mask = np.isfinite(labels)
        labels = np.nan_to_num(labels, nan=0.0).astype(np.float32)
        return {
            "id": int(record.get("index", index)),
            "file_path": str(record["file_path"]),
            "feats": torch.from_numpy(features).float(),
            "vad_target": torch.from_numpy(labels).float(),
            "vad_mask": torch.from_numpy(mask),
        }

    def __len__(self):
        return len(self.records)

    def collator(self, samples):
        if not samples:
            return {}

        features = [sample["feats"] for sample in samples]
        sizes = [feature.shape[0] for feature in features]
        target_size = max(sizes)
        collated = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        padding_mask = torch.zeros(len(features), target_size, dtype=torch.bool)
        for index, (feature, size) in enumerate(zip(features, sizes)):
            collated[index, :size] = feature
            padding_mask[index, size:] = True

        return {
            "id": torch.LongTensor([sample["id"] for sample in samples]),
            "file_path": [sample["file_path"] for sample in samples],
            "net_input": {"feats": collated, "padding_mask": padding_mask},
            "vad_labels": torch.stack(
                [sample["vad_target"] for sample in samples]
            ),
            "vad_mask": torch.stack(
                [sample["vad_mask"] for sample in samples]
            ).bool(),
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
