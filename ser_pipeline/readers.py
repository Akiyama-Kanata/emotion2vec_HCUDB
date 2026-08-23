"""Readers for MSP-Podcast R1.10, HCUDB1, and IEMOCAP metadata."""

from __future__ import annotations

import csv
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .contracts import MANIFEST_SCHEMA_VERSION, dataset_contract, map_emotion
from .splits import (
    IEMOCAP_SPLIT_VERSION,
    MSP_SPLIT_VERSION,
    hcudb_split_for_speaker,
    load_hcudb_split,
    msp_split,
)


def _posix(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _base_record(
    *,
    dataset: str,
    utterance_id: str,
    audio_relpath: str,
    speaker_id: str,
    speaker_id_status: str,
    group_id: str,
    session_id: str,
    source_split: str,
    split: str | None,
    split_version: str,
    original_emotion: str,
    extra_reasons: list[str] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = map_emotion(dataset, original_emotion)
    reasons = list(decision.exclusion_reasons)
    reasons.extend(extra_reasons or [])
    included = decision.included and not reasons and split is not None
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_release": dataset_contract(dataset)["dataset_release"],
        "utterance_id": utterance_id,
        "audio_relpath": audio_relpath,
        "audio_sha256": None,
        "speaker_id": speaker_id,
        "speaker_id_status": speaker_id_status,
        "group_id": group_id,
        "session_id": session_id,
        "source_split": source_split,
        "split": split,
        "split_version": split_version,
        "original_emotion": decision.original_emotion,
        "mapped_emotion": decision.mapped_emotion,
        "class_index": decision.class_index,
        "mapping_version": decision.mapping_version,
        "included": included,
        "exclusion_reasons": reasons,
        "approximate_mapping": decision.approximate_mapping,
        "audio_size_bytes": None,
        "sample_rate_hz": None,
        "channels": None,
        "num_samples": None,
        "duration_seconds": None,
        "source_metadata": source_metadata or {},
    }


def _resolve_existing(root: Path, candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} not found under {root}")


def read_msp_podcast(root: str | Path) -> Iterator[dict[str, Any]]:
    dataset_root = Path(root)
    csv_path = _resolve_existing(
        dataset_root,
        [dataset_root / "Labels" / "labels_consensus.csv", dataset_root / "labels_consensus.csv"],
        "MSP labels_consensus.csv",
    )
    partition_path = dataset_root / "Partitions.txt"
    official_partitions: dict[str, str] | None = None
    if partition_path.is_file():
        official_partitions = {}
        with partition_path.open("r", encoding="utf-8-sig") as partition_source:
            for line_number, line in enumerate(partition_source, 1):
                if not line.strip():
                    continue
                parts = [value.strip() for value in line.split(";", 1)]
                if len(parts) != 2 or not all(parts):
                    raise ValueError(f"invalid MSP Partitions.txt line {line_number}")
                source_split, filename = parts
                if filename in official_partitions:
                    raise ValueError(f"duplicate MSP partition filename: {filename}")
                official_partitions[filename] = source_split
    seen_filenames: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"FileName", "EmoClass", "SpkrID", "Split_Set"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"MSP metadata is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            filename = raw["FileName"].strip()
            if filename in seen_filenames:
                raise ValueError(f"duplicate MSP metadata filename: {filename}")
            seen_filenames.add(filename)
            utterance_id = Path(filename).stem
            source_split = raw["Split_Set"].strip()
            if official_partitions is not None:
                if official_partitions.get(filename) != source_split:
                    raise ValueError(f"MSP Partitions.txt mismatch for {filename}")
            split = msp_split(source_split)
            speaker = raw["SpkrID"].strip()
            status = "unknown" if speaker.casefold() == "unknown" else "known"
            reasons: list[str] = []
            if source_split == "Test2":
                reasons.append("msp_test2_out_of_scope")
            if status == "unknown":
                reasons.append("unknown_speaker")
            podcast_id = "_".join(utterance_id.split("_")[:2])
            yield _base_record(
                dataset="msp_podcast",
                utterance_id=utterance_id,
                audio_relpath=_posix("Audio", filename),
                speaker_id=speaker,
                speaker_id_status=status,
                group_id=speaker if status == "known" else podcast_id,
                session_id=podcast_id,
                source_split=source_split,
                split=split,
                split_version=MSP_SPLIT_VERSION,
                original_emotion=raw["EmoClass"].strip(),
                extra_reasons=reasons,
                source_metadata=dict(raw),
            )
    if official_partitions is not None:
        if not official_partitions:
            raise ValueError("MSP Partitions.txt is empty")
        if set(official_partitions) != seen_filenames:
            raise ValueError("MSP Partitions.txt and labels_consensus.csv filename sets differ")


def _hcudb_dataset_root(root: Path) -> Path:
    if (root / "wav").is_dir() and (root / "eval").is_dir():
        return root
    if (root / "HCUDB1" / "wav").is_dir():
        return root / "HCUDB1"
    return root


def read_hcudb1(root: str | Path) -> Iterator[dict[str, Any]]:
    supplied_root = Path(root)
    dataset_root = _hcudb_dataset_root(supplied_root)
    csv_path = _resolve_existing(
        dataset_root,
        [
            dataset_root / "eval" / "collected_result(all).csv",
            dataset_root / "eval" / "collected_result(all)_normalized.csv",
        ],
        "HCUDB collected metadata",
    )
    split_config = load_hcudb_split()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"音声ファイル名", "話者ID", "演技感情"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"HCUDB metadata is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            filename = raw["音声ファイル名"].strip()
            speaker = raw["話者ID"].strip()
            split = hcudb_split_for_speaker(speaker, split_config)
            yield _base_record(
                dataset="hcudb1",
                utterance_id=Path(filename).stem,
                audio_relpath=_posix("wav", speaker, filename),
                speaker_id=speaker,
                speaker_id_status="known",
                group_id=speaker,
                session_id="",
                source_split="all",
                split=split,
                split_version=split_config["split_version"],
                original_emotion=raw["演技感情"].strip(),
                source_metadata=dict(raw),
            )


def _iemocap_dataset_root(root: Path) -> Path:
    if (root / "IEMOCAP正解ラベル.csv").is_file():
        return root
    candidates = list(root.glob("*/IEMOCAP正解ラベル.csv"))
    return candidates[0].parent if len(candidates) == 1 else root


def read_iemocap(root: str | Path) -> Iterator[dict[str, Any]]:
    supplied_root = Path(root)
    dataset_root = _iemocap_dataset_root(supplied_root)
    csv_path = _resolve_existing(
        dataset_root,
        [dataset_root / "IEMOCAP正解ラベル.csv", dataset_root / "labels.csv"],
        "IEMOCAP metadata CSV",
    )
    compact_layout = (dataset_root / "Session1" / "wav").is_dir()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"ID", "Session", "emo"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"IEMOCAP metadata is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            utterance_id = raw["ID"].strip()
            session = raw["Session"].strip()
            match = re.search(r"(\d+)$", session)
            if match is None:
                raise ValueError(f"invalid IEMOCAP session: {session!r}")
            session_dir = f"Session{int(match.group(1))}"
            folder = utterance_id.rsplit("_", 1)[0]
            filename = raw.get("filename", "").strip() or f"{utterance_id}.wav"
            speaker = raw.get("SubSession", "").strip() or utterance_id[:6]
            yield _base_record(
                dataset="iemocap",
                utterance_id=utterance_id,
                audio_relpath=(
                    _posix(session_dir, "wav", folder, filename)
                    if compact_layout
                    else _posix(session_dir, "sentences", "wav", folder, filename)
                ),
                speaker_id=speaker,
                speaker_id_status="known",
                group_id=speaker,
                session_id=session,
                source_split="all_sessions",
                split="test",
                split_version=IEMOCAP_SPLIT_VERSION,
                original_emotion=raw["emo"].strip(),
                source_metadata=dict(raw),
            )


READERS = {
    "msp_podcast": read_msp_podcast,
    "hcudb1": read_hcudb1,
    "iemocap": read_iemocap,
}


def read_dataset(dataset: str, root: str | Path) -> Iterator[dict[str, Any]]:
    normalized = str(dataset).strip().lower()
    try:
        reader = READERS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset!r}") from exc
    yield from reader(root)


def resolved_dataset_root(dataset: str, root: str | Path) -> Path:
    candidate = Path(root)
    if dataset == "hcudb1":
        return _hcudb_dataset_root(candidate)
    if dataset == "iemocap":
        return _iemocap_dataset_root(candidate)
    return candidate
