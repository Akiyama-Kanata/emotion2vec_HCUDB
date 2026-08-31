"""Create and validate the fixed MSP-Podcast missing-audio exclusion contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


MSP_EXCLUSION_SCHEMA_VERSION = "msp_missing_audio_exclusions_v1"
MSP_EXCLUSION_REASON = "msp_missing_audio_exclusion_approved_v1"
MSP_EXPECTED_EXCLUDED_COUNT = 874
MSP_EXPECTED_INCLUDED_COUNT = 25_111
MSP_EXPECTED_ELIGIBLE_COUNT = MSP_EXPECTED_EXCLUDED_COUNT + MSP_EXPECTED_INCLUDED_COUNT
MSP_EXPECTED_ORIGINAL_LABEL_COUNTS = {"A": 378, "D": 24, "H": 392, "S": 80}
MSP_EXPECTED_MAPPED_LABEL_COUNTS = {
    "anger": 378,
    "disgust": 24,
    "happy": 392,
    "sadness": 80,
}
MSP_EXPECTED_SOURCE_SPLIT_COUNTS = {"Development": 210, "Test1": 144, "Train": 520}
MSP_ORIGINAL_TO_MAPPED = {"A": "anger", "D": "disgust", "H": "happy", "S": "sadness"}
MSP_SOURCE_TO_SPLIT = {"Development": "validation", "Test1": "test", "Train": "train"}

_CONTRACT_FIELDS = {
    "schema_version",
    "dataset",
    "dataset_release",
    "exclusion_reason",
    "count",
    "expected_included_count",
    "counts",
    "records",
    "normalized_sha256",
}
_RECORD_FIELDS = {
    "filename",
    "utterance_id",
    "original_emotion",
    "mapped_emotion",
    "official_split",
    "split",
    "exclusion_reason",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalized_exclusion_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical UTF-8 JSON after omitting the self-referential hash field."""
    normalized = dict(payload)
    normalized.pop("normalized_sha256", None)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def _record_from_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("dataset") != "msp_podcast" or not bool(row.get("included")):
        raise ValueError(f"exclusion candidate is not an eligible MSP row: {row.get('utterance_id')}")
    filename = PurePosixPath(str(row["audio_relpath"])).name
    return {
        "filename": filename,
        "utterance_id": str(row["utterance_id"]),
        "original_emotion": str(row["original_emotion"]),
        "mapped_emotion": str(row["mapped_emotion"]),
        "official_split": str(row["source_split"]),
        "split": str(row["split"]),
        "exclusion_reason": MSP_EXCLUSION_REASON,
    }


def build_msp_missing_audio_exclusion_contract(
    missing_metadata_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic v1 contract from currently missing eligible rows."""
    records = sorted(
        (_record_from_metadata(row) for row in missing_metadata_rows),
        key=lambda row: (row["filename"].casefold(), row["filename"]),
    )
    payload: dict[str, Any] = {
        "schema_version": MSP_EXCLUSION_SCHEMA_VERSION,
        "dataset": "msp_podcast",
        "dataset_release": "R1.10",
        "exclusion_reason": MSP_EXCLUSION_REASON,
        "count": len(records),
        "expected_included_count": MSP_EXPECTED_INCLUDED_COUNT,
        "counts": {
            "mapped_emotion": dict(sorted(Counter(row["mapped_emotion"] for row in records).items())),
            "official_split": dict(sorted(Counter(row["official_split"] for row in records).items())),
            "original_emotion": dict(sorted(Counter(row["original_emotion"] for row in records).items())),
        },
        "records": records,
    }
    payload["normalized_sha256"] = normalized_exclusion_contract_sha256(payload)
    validate_msp_missing_audio_exclusion_contract(payload)
    return payload


def _require_expected_counts(name: str, actual: Any, expected: Mapping[str, int]) -> None:
    if actual != dict(expected):
        raise ValueError(f"MSP exclusion contract {name} mismatch: expected {dict(expected)}, got {actual}")


def validate_msp_missing_audio_exclusion_contract(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate schema, fixed counts, ordering, uniqueness, and normalized SHA-256."""
    if set(payload) != _CONTRACT_FIELDS:
        raise ValueError(f"MSP exclusion contract fields mismatch: {sorted(set(payload) ^ _CONTRACT_FIELDS)}")
    if payload.get("schema_version") != MSP_EXCLUSION_SCHEMA_VERSION:
        raise ValueError("MSP exclusion contract schema_version mismatch")
    if payload.get("dataset") != "msp_podcast" or payload.get("dataset_release") != "R1.10":
        raise ValueError("MSP exclusion contract dataset identity mismatch")
    if payload.get("exclusion_reason") != MSP_EXCLUSION_REASON:
        raise ValueError("MSP exclusion contract reason mismatch")
    if payload.get("count") != MSP_EXPECTED_EXCLUDED_COUNT:
        raise ValueError(
            f"MSP exclusion contract count mismatch: expected {MSP_EXPECTED_EXCLUDED_COUNT}, "
            f"got {payload.get('count')}"
        )
    if payload.get("expected_included_count") != MSP_EXPECTED_INCLUDED_COUNT:
        raise ValueError("MSP exclusion contract expected included count mismatch")

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != MSP_EXPECTED_EXCLUDED_COUNT:
        raise ValueError("MSP exclusion contract records count mismatch")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ValueError(f"MSP exclusion contract record fields mismatch at index {index}")
        original = record.get("original_emotion")
        official_split = record.get("official_split")
        if record.get("mapped_emotion") != MSP_ORIGINAL_TO_MAPPED.get(str(original)):
            raise ValueError(f"MSP exclusion contract mapped label mismatch: {record.get('utterance_id')}")
        if record.get("split") != MSP_SOURCE_TO_SPLIT.get(str(official_split)):
            raise ValueError(f"MSP exclusion contract official split mismatch: {record.get('utterance_id')}")
        if record.get("exclusion_reason") != MSP_EXCLUSION_REASON:
            raise ValueError(f"MSP exclusion contract record reason mismatch: {record.get('utterance_id')}")
        if not str(record.get("filename", "")) or not str(record.get("utterance_id", "")):
            raise ValueError(f"MSP exclusion contract has an empty filename or utterance ID at index {index}")

    expected_order = sorted(records, key=lambda row: (row["filename"].casefold(), row["filename"]))
    if records != expected_order:
        raise ValueError("MSP exclusion contract records must be sorted by filename")
    ids = [str(record["utterance_id"]) for record in records]
    filenames = [str(record["filename"]).casefold() for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("MSP exclusion contract has duplicate utterance IDs")
    if len(filenames) != len(set(filenames)):
        raise ValueError("MSP exclusion contract has duplicate filenames")

    counts = payload.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"original_emotion", "mapped_emotion", "official_split"}:
        raise ValueError("MSP exclusion contract counts fields mismatch")
    _require_expected_counts("original label counts", counts["original_emotion"], MSP_EXPECTED_ORIGINAL_LABEL_COUNTS)
    _require_expected_counts("mapped label counts", counts["mapped_emotion"], MSP_EXPECTED_MAPPED_LABEL_COUNTS)
    _require_expected_counts("official split counts", counts["official_split"], MSP_EXPECTED_SOURCE_SPLIT_COUNTS)
    _require_expected_counts(
        "record original label counts",
        dict(sorted(Counter(record["original_emotion"] for record in records).items())),
        MSP_EXPECTED_ORIGINAL_LABEL_COUNTS,
    )
    _require_expected_counts(
        "record mapped label counts",
        dict(sorted(Counter(record["mapped_emotion"] for record in records).items())),
        MSP_EXPECTED_MAPPED_LABEL_COUNTS,
    )
    _require_expected_counts(
        "record official split counts",
        dict(sorted(Counter(record["official_split"] for record in records).items())),
        MSP_EXPECTED_SOURCE_SPLIT_COUNTS,
    )

    stored_sha256 = str(payload.get("normalized_sha256", "")).lower()
    actual_sha256 = normalized_exclusion_contract_sha256(payload)
    if not _SHA256_PATTERN.fullmatch(stored_sha256) or stored_sha256 != actual_sha256:
        raise ValueError("MSP exclusion contract normalized SHA-256 mismatch")
    if expected_sha256 is not None:
        approved_sha256 = str(expected_sha256).strip().lower()
        if not _SHA256_PATTERN.fullmatch(approved_sha256):
            raise ValueError("approved MSP exclusion SHA-256 must be 64 lowercase hexadecimal characters")
        if approved_sha256 != actual_sha256:
            raise ValueError("approved MSP exclusion SHA-256 mismatch")

    return {
        "schema_version": MSP_EXCLUSION_SCHEMA_VERSION,
        "normalized_sha256": actual_sha256,
        "count": MSP_EXPECTED_EXCLUDED_COUNT,
        "expected_included_count": MSP_EXPECTED_INCLUDED_COUNT,
        "counts": dict(counts),
    }


def load_msp_missing_audio_exclusion_contract(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a JSON contract and return its payload and validation report."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MSP exclusion contract JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MSP exclusion contract must be a JSON object")
    report = validate_msp_missing_audio_exclusion_contract(payload, expected_sha256=expected_sha256)
    return payload, report


def write_msp_missing_audio_exclusion_contract(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a validated contract with stable formatting."""
    validate_msp_missing_audio_exclusion_contract(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(output)
    return output


def reconcile_msp_exclusion_contract(
    metadata_rows: Iterable[Mapping[str, Any]],
    missing_eligible_ids: Iterable[str],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact agreement between an approved contract, metadata, and current absence."""
    rows = list(metadata_rows)
    eligible = {str(row["utterance_id"]): row for row in rows if bool(row.get("included"))}
    if len(eligible) != MSP_EXPECTED_ELIGIBLE_COUNT:
        raise ValueError(
            f"MSP eligible metadata count mismatch: expected {MSP_EXPECTED_ELIGIBLE_COUNT}, got {len(eligible)}"
        )
    records = list(payload["records"])
    contract_by_id = {str(record["utterance_id"]): record for record in records}
    unknown = sorted(set(contract_by_id) - set(eligible))
    if unknown:
        raise ValueError(f"MSP exclusion contract target is not eligible metadata: {unknown[:5]}")

    comparable_fields = {
        "filename": lambda row: PurePosixPath(str(row["audio_relpath"])).name,
        "original_emotion": lambda row: str(row["original_emotion"]),
        "mapped_emotion": lambda row: str(row["mapped_emotion"]),
        "official_split": lambda row: str(row["source_split"]),
        "split": lambda row: str(row["split"]),
    }
    for identifier, record in contract_by_id.items():
        metadata = eligible[identifier]
        for field, getter in comparable_fields.items():
            if record[field] != getter(metadata):
                raise ValueError(f"MSP exclusion contract metadata mismatch for {field}: {identifier}")

    missing = {str(identifier) for identifier in missing_eligible_ids}
    contract_ids = set(contract_by_id)
    recovered = sorted(contract_ids - missing)
    if recovered:
        raise ValueError(f"MSP exclusion contract is stale because target audio is now available: {recovered[:5]}")
    unapproved = sorted(missing - contract_ids)
    if unapproved:
        raise ValueError(f"MSP eligible audio is missing outside the approved exclusion contract: {unapproved[:5]}")
    if len(eligible) - len(contract_ids) != MSP_EXPECTED_INCLUDED_COUNT:
        raise ValueError("MSP final included count does not equal 25,111")
    return {
        "eligible_metadata": len(eligible),
        "approved_missing": len(contract_ids),
        "unapproved_missing": 0,
        "final_included": MSP_EXPECTED_INCLUDED_COUNT,
    }


def manifest_exclusion_contract_signature(records: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Validate and summarize contract provenance embedded in manifest rows."""
    rows = list(records)
    provenance_rows = [row for row in rows if row.get("exclusion_contract_sha256") is not None]
    if not provenance_rows:
        if any(row.get("exclusion_contract_schema_version") is not None for row in rows):
            raise ValueError("manifest exclusion contract provenance is incomplete")
        return None
    if len(provenance_rows) != len(rows):
        raise ValueError("manifest exclusion contract provenance must be present on every row")
    hashes = {str(row.get("exclusion_contract_sha256", "")) for row in rows}
    schemas = {str(row.get("exclusion_contract_schema_version", "")) for row in rows}
    if len(hashes) != 1 or len(schemas) != 1:
        raise ValueError("manifest exclusion contract provenance is inconsistent")
    contract_sha256 = next(iter(hashes))
    if not _SHA256_PATTERN.fullmatch(contract_sha256):
        raise ValueError("manifest exclusion contract SHA-256 is invalid")
    if schemas != {MSP_EXCLUSION_SCHEMA_VERSION}:
        raise ValueError("manifest exclusion contract schema version mismatch")
    if {str(row.get("dataset")) for row in rows} != {"msp_podcast"}:
        raise ValueError("MSP exclusion contract provenance is valid only for an MSP manifest")

    excluded = [row for row in rows if MSP_EXCLUSION_REASON in row.get("exclusion_reasons", [])]
    if len(excluded) != MSP_EXPECTED_EXCLUDED_COUNT:
        raise ValueError("manifest approved MSP exclusion count mismatch")
    if any(bool(row.get("included")) for row in excluded):
        raise ValueError("manifest approved MSP exclusions must have included=false")
    if sum(bool(row.get("included")) for row in rows) != MSP_EXPECTED_INCLUDED_COUNT:
        raise ValueError("manifest final included count does not equal 25,111")
    _require_expected_counts(
        "manifest original label counts",
        dict(sorted(Counter(str(row["original_emotion"]) for row in excluded).items())),
        MSP_EXPECTED_ORIGINAL_LABEL_COUNTS,
    )
    _require_expected_counts(
        "manifest mapped label counts",
        dict(sorted(Counter(str(row["mapped_emotion"]) for row in excluded).items())),
        MSP_EXPECTED_MAPPED_LABEL_COUNTS,
    )
    _require_expected_counts(
        "manifest official split counts",
        dict(sorted(Counter(str(row["source_split"]) for row in excluded).items())),
        MSP_EXPECTED_SOURCE_SPLIT_COUNTS,
    )
    return {
        "schema_version": MSP_EXCLUSION_SCHEMA_VERSION,
        "normalized_sha256": contract_sha256,
        "count": MSP_EXPECTED_EXCLUDED_COUNT,
        "counts": {
            "mapped_emotion": dict(MSP_EXPECTED_MAPPED_LABEL_COUNTS),
            "official_split": dict(MSP_EXPECTED_SOURCE_SPLIT_COUNTS),
            "original_emotion": dict(MSP_EXPECTED_ORIGINAL_LABEL_COUNTS),
        },
        "final_included": MSP_EXPECTED_INCLUDED_COUNT,
    }


__all__ = [
    "MSP_EXCLUSION_REASON",
    "MSP_EXCLUSION_SCHEMA_VERSION",
    "MSP_EXPECTED_EXCLUDED_COUNT",
    "MSP_EXPECTED_INCLUDED_COUNT",
    "build_msp_missing_audio_exclusion_contract",
    "load_msp_missing_audio_exclusion_contract",
    "manifest_exclusion_contract_signature",
    "normalized_exclusion_contract_sha256",
    "reconcile_msp_exclusion_contract",
    "validate_msp_missing_audio_exclusion_contract",
    "write_msp_missing_audio_exclusion_contract",
]
