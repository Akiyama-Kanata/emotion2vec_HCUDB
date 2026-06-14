import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FEATURE_DIM = 768


def load_vad_dataset(prefix, min_length=1, max_length=None):
    """Load frame-level emotion2vec features with utterance-level VA/VAD labels."""
    feats = np.load(prefix + ".npy")
    _validate_features(feats, prefix + ".npy")

    lengths = _load_lengths(prefix + ".lengths")
    if int(np.sum(lengths, dtype=np.int64)) != feats.shape[0]:
        raise ValueError(
            f"sum of lengths ({int(np.sum(lengths, dtype=np.int64))}) does not "
            f"match feature frames ({feats.shape[0]})"
        )

    utt_ids, targets, target_dim = _load_vad_labels(prefix + ".vad")
    if len(utt_ids) != len(lengths):
        raise ValueError(
            f"number of VAD rows ({len(utt_ids)}) does not match lengths "
            f"({len(lengths)})"
        )

    sizes_all = np.asarray(lengths, dtype=np.int64)
    offsets_all = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(sizes_all, dtype=np.int64)[:-1]]
    )

    keep = []
    for i, size in enumerate(sizes_all):
        if size >= min_length and (max_length is None or size <= max_length):
            keep.append(i)

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
