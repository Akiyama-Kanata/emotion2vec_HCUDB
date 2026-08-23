"""Fixed split assignment and leakage validation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Any


MSP_SPLIT_VERSION = "msp_podcast_r1_10_official_v1"
IEMOCAP_SPLIT_VERSION = "iemocap_all_sessions_external_test_v1"
HCUDB_SPLIT_PATH = Path(__file__).with_name("config") / "hcudb1_speaker_split.v1.json"
MSP_SOURCE_TO_SPLIT = {"Train": "train", "Development": "validation", "Test1": "test"}
VALID_SPLITS = ("train", "validation", "test")


def derive_hcudb_split(speakers: Iterable[str], seed: int = 42) -> dict[str, list[str]]:
    values = sorted(set(str(speaker) for speaker in speakers))
    female = sorted(speaker for speaker in values if speaker.startswith("F"))
    male = sorted(speaker for speaker in values if speaker.startswith("M"))
    if len(female) != 8 or len(male) != 6:
        raise ValueError("HCUDB split derivation requires 8 female and 6 male speakers")
    generator = random.Random(seed)
    generator.shuffle(female)
    generator.shuffle(male)
    return {
        "train": sorted(female[2:] + male[2:]),
        "validation": [female[0], male[0]],
        "test": [female[1], male[1]],
    }


def load_hcudb_split(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else HCUDB_SPLIT_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    assigned = [speaker for split in VALID_SPLITS for speaker in payload["splits"].get(split, [])]
    if len(assigned) != 14 or len(set(assigned)) != 14:
        raise ValueError("HCUDB split must assign exactly 14 unique speakers")
    if [len(payload["splits"][name]) for name in VALID_SPLITS] != [10, 2, 2]:
        raise ValueError("HCUDB split sizes must be 10/2/2")
    derived = derive_hcudb_split(assigned, int(payload.get("seed", -1)))
    configured = {split: sorted(payload["splits"][split]) for split in VALID_SPLITS}
    if derived != configured:
        raise ValueError("HCUDB fixed split does not match the documented random.Random procedure")
    return payload


def hcudb_split_for_speaker(speaker_id: str, config: dict[str, Any] | None = None) -> str:
    payload = load_hcudb_split() if config is None else config
    matches = [split for split, speakers in payload["splits"].items() if speaker_id in speakers]
    if len(matches) != 1:
        raise ValueError(f"HCUDB speaker is not assigned exactly once: {speaker_id!r}")
    return matches[0]


def msp_split(source_split: str) -> str | None:
    if source_split == "Test2":
        return None
    try:
        return MSP_SOURCE_TO_SPLIT[source_split]
    except KeyError as exc:
        raise ValueError(f"unknown MSP official split: {source_split!r}") from exc


def _duplicate_values(rows: list[Mapping[str, Any]], field: str) -> list[str]:
    split_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            split_sets[str(value)].add(str(row["split"]))
    return sorted(value for value, splits in split_sets.items() if len(splits) > 1)


def validate_split_integrity(
    records: Iterable[Mapping[str, Any]],
    *,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    rows = [row for row in records if bool(row.get("included"))]
    if not rows:
        raise ValueError("manifest has no included rows")
    datasets = {str(row["dataset"]) for row in rows}
    if len(datasets) != 1:
        raise ValueError("split validation expects one dataset at a time")
    dataset = next(iter(datasets))
    ids = [str(row["utterance_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate included utterance_id detected")
    unknown = [row["utterance_id"] for row in rows if row.get("speaker_id_status") == "unknown"]
    if unknown:
        raise ValueError("included rows must not contain Unknown speakers")

    expected = {"test"} if dataset == "iemocap" else set(VALID_SPLITS)
    counts = {split: sum(row["split"] == split for row in rows) for split in expected}
    if require_nonempty and any(count == 0 for count in counts.values()):
        raise ValueError(f"empty required split detected: {counts}")
    unexpected = sorted({str(row["split"]) for row in rows} - expected)
    if unexpected:
        raise ValueError(f"unexpected split values: {unexpected}")

    for field, description in (
        ("speaker_id", "speaker"),
        ("utterance_id", "utterance"),
        ("audio_sha256", "audio hash"),
    ):
        duplicates = _duplicate_values(rows, field)
        if duplicates:
            raise ValueError(f"{description} leakage across splits: {duplicates[:5]}")

    if dataset == "msp_podcast":
        for row in rows:
            expected_split = msp_split(str(row["source_split"]))
            if expected_split != row["split"]:
                raise ValueError(f"MSP official split mismatch for {row['utterance_id']}")
            if row["split_version"] != MSP_SPLIT_VERSION:
                raise ValueError("MSP split_version mismatch")
    elif dataset == "hcudb1":
        config = load_hcudb_split()
        for row in rows:
            if hcudb_split_for_speaker(str(row["speaker_id"]), config) != row["split"]:
                raise ValueError(f"HCUDB speaker assignment mismatch for {row['speaker_id']}")
            if row["split_version"] != config["split_version"]:
                raise ValueError("HCUDB split_version mismatch")
    elif dataset == "iemocap":
        if any(row["split_version"] != IEMOCAP_SPLIT_VERSION for row in rows):
            raise ValueError("IEMOCAP split_version mismatch")

    return {"dataset": dataset, "included": len(rows), "split_counts": counts, "status": "ok"}
