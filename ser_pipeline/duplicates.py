"""Audit exact MSP-Podcast audio duplicates and validate approved exclusions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audio import inspect_audio


MSP_DUPLICATE_AUDIT_SCHEMA_VERSION = "msp_audio_duplicate_audit_v1"
MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION = "msp_audio_duplicate_exclusions_v1"
MSP_DUPLICATE_EXCLUSION_REASON = "msp_audio_duplicate_exclusion_approved_v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_AUDIT_FIELDS = {
    "schema_version",
    "dataset",
    "dataset_release",
    "missing_audio_exclusion_contract",
    "method",
    "records",
    "duplicate_groups",
    "summary",
    "normalized_sha256",
}
_AUDIT_RECORD_FIELDS = {
    "utterance_id",
    "audio_relpath",
    "source_split",
    "split",
    "speaker_id",
    "original_emotion",
    "mapped_emotion",
    "byte_sha256",
    "audio_size_bytes",
    "sample_rate_hz",
    "channels",
    "num_frames",
    "decoded_waveform_candidate",
    "decoded_waveform_sha256",
}
_GROUP_FIELDS = {
    "group_id",
    "member_ids",
    "match_types",
    "byte_exact_hashes",
    "decoded_waveform_hashes",
    "splits",
    "cross_split",
    "speaker_mismatch",
    "label_mismatch",
}
_CONTRACT_FIELDS = {
    "schema_version",
    "dataset",
    "dataset_release",
    "audit_schema_version",
    "audit_normalized_sha256",
    "missing_audio_exclusion_contract",
    "exclusion_reason",
    "count",
    "post_exclusion_counts",
    "records",
    "normalized_sha256",
}
_CONTRACT_RECORD_FIELDS = {
    "utterance_id",
    "audio_relpath",
    "source_split",
    "split",
    "speaker_id",
    "original_emotion",
    "mapped_emotion",
    "byte_sha256",
    "decoded_waveform_sha256",
    "duplicate_group_id",
    "exclusion_reason",
}
_MANIFEST_PROVENANCE_FIELDS = (
    "duplicate_audit_schema_version",
    "duplicate_audit_sha256",
    "duplicate_audit_target_count",
    "duplicate_exclusion_contract_schema_version",
    "duplicate_exclusion_contract_sha256",
)
_CSV_FIELDS = (
    "group_id",
    "utterance_id",
    "audio_relpath",
    "source_split",
    "split",
    "speaker_id",
    "original_emotion",
    "mapped_emotion",
    "byte_sha256",
    "decoded_waveform_sha256",
    "cross_split",
    "speaker_mismatch",
    "label_mismatch",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalized_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("normalized_sha256", None)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def normalized_duplicate_audit_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical audit hash without its self-referential field."""
    return _normalized_sha256(payload)


def normalized_duplicate_exclusion_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical duplicate-exclusion contract hash."""
    return _normalized_sha256(payload)


def _require_sha256(value: Any, description: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{description} must be 64 lowercase hexadecimal characters")
    return normalized


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    relpath = str(record["audio_relpath"])
    return relpath.casefold(), relpath, str(record["utterance_id"])


def _shape_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(record["sample_rate_hz"]), int(record["channels"]), int(record["num_frames"])


def _audit_method() -> dict[str, Any]:
    return {
        "byte_fingerprint": {
            "algorithm": "sha256",
            "input": "complete_file_bytes",
        },
        "decoded_waveform_candidate_rule": "equal_sample_rate_channels_and_frames",
        "decoded_waveform_fingerprint": {
            "algorithm": "sha256",
            "decoder": "soundfile",
            "dtype": "little_endian_float32",
            "always_2d": True,
            "shape_fields": ["sample_rate_hz", "num_frames", "channels"],
            "storage_order": "C_interleaved_frames_channels",
        },
        "resampling": False,
        "tolerance": None,
        "similarity_threshold": None,
        "approximate_duplicates_in_scope": False,
    }


def _decoded_waveform_sha256(path: Path, expected_shape: tuple[int, int, int]) -> str:
    """Hash exact decoded samples as C-order little-endian float32 plus audio shape."""
    import soundfile as sf

    try:
        waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"unreadable audio during duplicate audit: {path}") from exc
    expected_rate, expected_channels, expected_frames = expected_shape
    if int(sample_rate) != expected_rate or tuple(waveform.shape) != (expected_frames, expected_channels):
        raise ValueError(f"decoded audio shape changed during duplicate audit: {path}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"decoded audio contains non-finite values: {path}")
    little_endian = np.asarray(waveform, dtype=np.dtype("<f4"), order="C")
    header = {
        "sample_rate_hz": expected_rate,
        "shape": [expected_frames, expected_channels],
        "dtype": "<f4",
        "storage_order": "C_interleaved_frames_channels",
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\n")
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _duplicate_groups(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(record["utterance_id"]): record for record in records}
    parent = {identifier: identifier for identifier in by_id}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    byte_buckets: dict[str, list[str]] = defaultdict(list)
    waveform_buckets: dict[str, list[str]] = defaultdict(list)
    for record in records:
        identifier = str(record["utterance_id"])
        byte_buckets[str(record["byte_sha256"])].append(identifier)
        waveform_sha = record.get("decoded_waveform_sha256")
        if waveform_sha is not None:
            waveform_buckets[str(waveform_sha)].append(identifier)
    for buckets in (byte_buckets, waveform_buckets):
        for members in buckets.values():
            if len(members) > 1:
                for member in members[1:]:
                    union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    for identifier in by_id:
        components[find(identifier)].append(identifier)

    groups: list[dict[str, Any]] = []
    for member_ids in components.values():
        if len(member_ids) < 2:
            continue
        members = sorted((by_id[identifier] for identifier in member_ids), key=_record_sort_key)
        ordered_ids = [str(member["utterance_id"]) for member in members]
        byte_hashes = sorted(
            fingerprint
            for fingerprint, count in Counter(str(member["byte_sha256"]) for member in members).items()
            if count > 1
        )
        waveform_hashes = sorted(
            fingerprint
            for fingerprint, count in Counter(
                str(member["decoded_waveform_sha256"])
                for member in members
                if member.get("decoded_waveform_sha256") is not None
            ).items()
            if count > 1
        )
        match_types = []
        if byte_hashes:
            match_types.append("byte_exact")
        if waveform_hashes:
            match_types.append("decoded_waveform_exact")
        splits = sorted({str(member["split"]) for member in members})
        group_digest = hashlib.sha256(("\n".join(sorted(ordered_ids)) + "\n").encode("utf-8")).hexdigest()
        groups.append(
            {
                "group_id": f"dup-{group_digest[:16]}",
                "member_ids": ordered_ids,
                "match_types": match_types,
                "byte_exact_hashes": byte_hashes,
                "decoded_waveform_hashes": waveform_hashes,
                "splits": splits,
                "cross_split": len(splits) > 1,
                "speaker_mismatch": len({str(member["speaker_id"]) for member in members}) > 1,
                "label_mismatch": len({str(member["mapped_emotion"]) for member in members}) > 1,
            }
        )
    return sorted(groups, key=lambda group: (group["member_ids"][0], group["group_id"]))


def _audit_summary(records: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    duplicate_ids = {identifier for group in groups for identifier in group["member_ids"]}
    byte_groups = sum("byte_exact" in group["match_types"] for group in groups)
    waveform_groups = sum("decoded_waveform_exact" in group["match_types"] for group in groups)
    cross_groups = sum(bool(group["cross_split"]) for group in groups)
    return {
        "target_files": len(records),
        "decoded_waveform_candidates": sum(bool(record["decoded_waveform_candidate"]) for record in records),
        "duplicate_groups": len(groups),
        "duplicate_members": len(duplicate_ids),
        "byte_exact_groups": byte_groups,
        "decoded_waveform_exact_groups": waveform_groups,
        "within_split_groups": len(groups) - cross_groups,
        "cross_split_groups": cross_groups,
        "speaker_mismatch_groups": sum(bool(group["speaker_mismatch"]) for group in groups),
        "label_mismatch_groups": sum(bool(group["label_mismatch"]) for group in groups),
    }


def build_msp_audio_duplicate_audit(
    metadata_rows: Iterable[Mapping[str, Any]],
    audio_paths: Mapping[str, str | Path],
    *,
    missing_exclusion_contract_schema_version: str,
    missing_exclusion_contract_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic exact-duplicate audit without changing audio or manifests."""
    missing_sha256 = _require_sha256(
        missing_exclusion_contract_sha256,
        "missing-audio exclusion contract SHA-256",
    )
    rows = list(metadata_rows)
    identifiers = [str(row.get("utterance_id", "")) for row in rows]
    if not rows or any(not identifier for identifier in identifiers):
        raise ValueError("MSP duplicate audit requires non-empty utterance IDs")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("MSP duplicate audit metadata has duplicate utterance IDs")
    if set(audio_paths) != set(identifiers):
        raise ValueError("MSP duplicate audit audio paths must match target utterance IDs exactly")

    records: list[dict[str, Any]] = []
    path_by_id = {identifier: Path(path) for identifier, path in audio_paths.items()}
    for row in rows:
        identifier = str(row["utterance_id"])
        if (
            row.get("dataset") != "msp_podcast"
            or not bool(row.get("included"))
            or row.get("speaker_id_status") != "known"
            or row.get("split") not in {"train", "validation", "test"}
        ):
            raise ValueError(f"MSP duplicate audit row is outside the research target: {identifier}")
        path = path_by_id[identifier]
        if not path.is_file():
            raise ValueError(f"MSP duplicate audit target audio is missing: {identifier}")
        audio = inspect_audio(path, compute_sha256=True)
        records.append(
            {
                "utterance_id": identifier,
                "audio_relpath": str(row["audio_relpath"]),
                "source_split": str(row["source_split"]),
                "split": str(row["split"]),
                "speaker_id": str(row["speaker_id"]),
                "original_emotion": str(row["original_emotion"]),
                "mapped_emotion": str(row["mapped_emotion"]),
                "byte_sha256": str(audio["audio_sha256"]),
                "audio_size_bytes": int(audio["audio_size_bytes"]),
                "sample_rate_hz": int(audio["sample_rate_hz"]),
                "channels": int(audio["channels"]),
                "num_frames": int(audio["num_samples"]),
                "decoded_waveform_candidate": False,
                "decoded_waveform_sha256": None,
            }
        )
    records.sort(key=_record_sort_key)

    shape_buckets: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        shape_buckets[_shape_key(record)].append(record)
    for shape, candidates in shape_buckets.items():
        if len(candidates) < 2:
            continue
        for record in candidates:
            record["decoded_waveform_candidate"] = True
            record["decoded_waveform_sha256"] = _decoded_waveform_sha256(
                path_by_id[str(record["utterance_id"])],
                shape,
            )

    groups = _duplicate_groups(records)
    payload: dict[str, Any] = {
        "schema_version": MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
        "dataset": "msp_podcast",
        "dataset_release": "R1.10",
        "missing_audio_exclusion_contract": {
            "schema_version": str(missing_exclusion_contract_schema_version),
            "normalized_sha256": missing_sha256,
        },
        "method": _audit_method(),
        "records": records,
        "duplicate_groups": groups,
        "summary": _audit_summary(records, groups),
    }
    payload["normalized_sha256"] = normalized_duplicate_audit_sha256(payload)
    validate_msp_audio_duplicate_audit(payload)
    return payload


def validate_msp_audio_duplicate_audit(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate audit schema, deterministic grouping, summaries, and canonical SHA-256."""
    if set(payload) != _AUDIT_FIELDS:
        raise ValueError(f"MSP duplicate audit fields mismatch: {sorted(set(payload) ^ _AUDIT_FIELDS)}")
    if payload.get("schema_version") != MSP_DUPLICATE_AUDIT_SCHEMA_VERSION:
        raise ValueError("MSP duplicate audit schema_version mismatch")
    if payload.get("dataset") != "msp_podcast" or payload.get("dataset_release") != "R1.10":
        raise ValueError("MSP duplicate audit dataset identity mismatch")
    missing_contract = payload.get("missing_audio_exclusion_contract")
    if not isinstance(missing_contract, dict) or set(missing_contract) != {"schema_version", "normalized_sha256"}:
        raise ValueError("MSP duplicate audit missing-audio contract reference is invalid")
    _require_sha256(missing_contract["normalized_sha256"], "missing-audio exclusion contract SHA-256")

    method = payload.get("method")
    if method != _audit_method():
        raise ValueError("MSP duplicate audit method contract mismatch")

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("MSP duplicate audit records must be a non-empty list")
    identifiers: list[str] = []
    relative_paths: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _AUDIT_RECORD_FIELDS:
            raise ValueError(f"MSP duplicate audit record fields mismatch at index {index}")
        identifier = str(record.get("utterance_id", ""))
        identifiers.append(identifier)
        relpath = str(record.get("audio_relpath", ""))
        relative_paths.append(relpath.casefold())
        if not identifier or not relpath:
            raise ValueError(f"MSP duplicate audit record has an empty identity at index {index}")
        if PurePosixPath(relpath).is_absolute() or PureWindowsPath(relpath).is_absolute():
            raise ValueError(f"MSP duplicate audit audio_relpath must be relative: {identifier}")
        _require_sha256(record.get("byte_sha256"), f"byte SHA-256 for {identifier}")
        if any(int(record[field]) <= 0 for field in ("audio_size_bytes", "sample_rate_hz", "channels", "num_frames")):
            raise ValueError(f"MSP duplicate audit record has invalid audio shape: {identifier}")
    if records != sorted(records, key=_record_sort_key):
        raise ValueError("MSP duplicate audit records are not deterministically ordered")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("MSP duplicate audit has duplicate utterance IDs")
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("MSP duplicate audit has duplicate relative audio paths")

    shape_counts = Counter(_shape_key(record) for record in records)
    for record in records:
        expected_candidate = shape_counts[_shape_key(record)] > 1
        identifier = str(record["utterance_id"])
        if record.get("decoded_waveform_candidate") is not expected_candidate:
            raise ValueError(f"MSP duplicate audit waveform candidate flag mismatch: {identifier}")
        waveform_sha = record.get("decoded_waveform_sha256")
        if expected_candidate:
            _require_sha256(waveform_sha, f"decoded waveform SHA-256 for {identifier}")
        elif waveform_sha is not None:
            raise ValueError(f"MSP duplicate audit decoded a non-candidate: {identifier}")

    groups = payload.get("duplicate_groups")
    if not isinstance(groups, list):
        raise ValueError("MSP duplicate audit groups must be a list")
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != _GROUP_FIELDS:
            raise ValueError(f"MSP duplicate audit group fields mismatch at index {index}")
    expected_groups = _duplicate_groups(records)
    if groups != expected_groups:
        raise ValueError("MSP duplicate audit groups do not match the stored fingerprints")
    expected_summary = _audit_summary(records, groups)
    if payload.get("summary") != expected_summary:
        raise ValueError("MSP duplicate audit summary mismatch")

    actual_sha256 = normalized_duplicate_audit_sha256(payload)
    stored_sha256 = _require_sha256(payload.get("normalized_sha256"), "MSP duplicate audit normalized SHA-256")
    if stored_sha256 != actual_sha256:
        raise ValueError("MSP duplicate audit normalized SHA-256 mismatch")
    if expected_sha256 is not None and _require_sha256(expected_sha256, "approved audit SHA-256") != actual_sha256:
        raise ValueError("approved MSP duplicate audit SHA-256 mismatch")
    return {
        "schema_version": MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
        "normalized_sha256": actual_sha256,
        "target_count": len(records),
        "summary": dict(expected_summary),
        "missing_audio_exclusion_contract": dict(missing_contract),
    }


def load_msp_audio_duplicate_audit(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate an msp_audio_duplicate_audit_v1 JSON file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MSP duplicate audit JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MSP duplicate audit must be a JSON object")
    return payload, validate_msp_audio_duplicate_audit(payload, expected_sha256=expected_sha256)


def write_msp_audio_duplicate_audit(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a validated duplicate audit with stable formatting."""
    validate_msp_audio_duplicate_audit(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(output)
    return output


def write_msp_audio_duplicate_candidates_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write only duplicate members to a deterministic UTF-8 CSV."""
    validate_msp_audio_duplicate_audit(payload)
    by_id = {str(record["utterance_id"]): record for record in payload["records"]}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for group in payload["duplicate_groups"]:
            for identifier in group["member_ids"]:
                record = by_id[str(identifier)]
                writer.writerow(
                    {
                        "group_id": group["group_id"],
                        "utterance_id": identifier,
                        "audio_relpath": record["audio_relpath"],
                        "source_split": record["source_split"],
                        "split": record["split"],
                        "speaker_id": record["speaker_id"],
                        "original_emotion": record["original_emotion"],
                        "mapped_emotion": record["mapped_emotion"],
                        "byte_sha256": record["byte_sha256"],
                        "decoded_waveform_sha256": record["decoded_waveform_sha256"],
                        "cross_split": str(bool(group["cross_split"])).lower(),
                        "speaker_mismatch": str(bool(group["speaker_mismatch"])).lower(),
                        "label_mismatch": str(bool(group["label_mismatch"])).lower(),
                    }
                )
    partial.replace(output)
    return output


def _group_by_member(audit_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for group in audit_payload["duplicate_groups"]:
        for identifier in group["member_ids"]:
            if str(identifier) in result:
                raise ValueError(f"MSP duplicate audit member belongs to multiple groups: {identifier}")
            result[str(identifier)] = group
    return result


def build_msp_audio_duplicate_exclusion_contract(
    audit_payload: Mapping[str, Any],
    approved_utterance_ids: Sequence[str],
) -> dict[str, Any]:
    """Build a contract only from explicit IDs present in duplicate candidates."""
    audit_report = validate_msp_audio_duplicate_audit(audit_payload)
    approved = [str(identifier).strip() for identifier in approved_utterance_ids]
    if any(not identifier for identifier in approved):
        raise ValueError("approved duplicate exclusion IDs must not be empty")
    if len(approved) != len(set(approved)):
        raise ValueError("approved duplicate exclusion IDs contain duplicates")
    group_by_member = _group_by_member(audit_payload)
    unknown = sorted(set(approved) - set(group_by_member))
    if unknown:
        raise ValueError(f"approved duplicate exclusion ID is not an audit candidate: {unknown[:5]}")

    approved_set = set(approved)
    unresolved: list[str] = []
    for group in audit_payload["duplicate_groups"]:
        remaining_splits = {
            str(record["split"])
            for record in audit_payload["records"]
            if record["utterance_id"] in group["member_ids"] and record["utterance_id"] not in approved_set
        }
        if bool(group["cross_split"]) and len(remaining_splits) > 1:
            unresolved.append(str(group["group_id"]))
    if unresolved:
        raise ValueError(f"unresolved cross-split duplicate groups remain: {unresolved[:5]}")

    by_id = {str(record["utterance_id"]): record for record in audit_payload["records"]}
    records = []
    for identifier in approved_set:
        audit_record = by_id[identifier]
        group = group_by_member[identifier]
        records.append(
            {
                "utterance_id": identifier,
                "audio_relpath": audit_record["audio_relpath"],
                "source_split": audit_record["source_split"],
                "split": audit_record["split"],
                "speaker_id": audit_record["speaker_id"],
                "original_emotion": audit_record["original_emotion"],
                "mapped_emotion": audit_record["mapped_emotion"],
                "byte_sha256": audit_record["byte_sha256"],
                "decoded_waveform_sha256": audit_record["decoded_waveform_sha256"],
                "duplicate_group_id": group["group_id"],
                "exclusion_reason": MSP_DUPLICATE_EXCLUSION_REASON,
            }
        )
    records.sort(key=_record_sort_key)
    remaining = [record for record in audit_payload["records"] if record["utterance_id"] not in approved_set]
    post_counts = {
        "audited_available": len(audit_payload["records"]),
        "approved_duplicate_exclusions": len(records),
        "final_included": len(remaining),
        "split_counts": dict(sorted(Counter(str(record["split"]) for record in remaining).items())),
    }
    payload: dict[str, Any] = {
        "schema_version": MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION,
        "dataset": "msp_podcast",
        "dataset_release": "R1.10",
        "audit_schema_version": MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
        "audit_normalized_sha256": audit_report["normalized_sha256"],
        "missing_audio_exclusion_contract": dict(audit_payload["missing_audio_exclusion_contract"]),
        "exclusion_reason": MSP_DUPLICATE_EXCLUSION_REASON,
        "count": len(records),
        "post_exclusion_counts": post_counts,
        "records": records,
    }
    payload["normalized_sha256"] = normalized_duplicate_exclusion_contract_sha256(payload)
    validate_msp_audio_duplicate_exclusion_contract(payload, audit_payload)
    return payload


def validate_msp_audio_duplicate_exclusion_contract(
    payload: Mapping[str, Any],
    audit_payload: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate approved IDs, audit linkage, residual splits, counts, and SHA-256."""
    audit_report = validate_msp_audio_duplicate_audit(audit_payload)
    if set(payload) != _CONTRACT_FIELDS:
        raise ValueError(f"MSP duplicate exclusion contract fields mismatch: {sorted(set(payload) ^ _CONTRACT_FIELDS)}")
    if payload.get("schema_version") != MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION:
        raise ValueError("MSP duplicate exclusion contract schema_version mismatch")
    if payload.get("dataset") != "msp_podcast" or payload.get("dataset_release") != "R1.10":
        raise ValueError("MSP duplicate exclusion contract dataset identity mismatch")
    if payload.get("audit_schema_version") != MSP_DUPLICATE_AUDIT_SCHEMA_VERSION:
        raise ValueError("MSP duplicate exclusion contract audit schema mismatch")
    if payload.get("audit_normalized_sha256") != audit_report["normalized_sha256"]:
        raise ValueError("MSP duplicate exclusion contract audit SHA-256 mismatch")
    if payload.get("missing_audio_exclusion_contract") != audit_payload["missing_audio_exclusion_contract"]:
        raise ValueError("MSP duplicate exclusion contract missing-audio provenance mismatch")
    if payload.get("exclusion_reason") != MSP_DUPLICATE_EXCLUSION_REASON:
        raise ValueError("MSP duplicate exclusion contract reason mismatch")

    records = payload.get("records")
    if not isinstance(records, list) or payload.get("count") != len(records):
        raise ValueError("MSP duplicate exclusion contract count mismatch")
    if records != sorted(records, key=_record_sort_key):
        raise ValueError("MSP duplicate exclusion contract records are not deterministically ordered")
    identifiers: list[str] = []
    audit_by_id = {str(record["utterance_id"]): record for record in audit_payload["records"]}
    group_by_member = _group_by_member(audit_payload)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _CONTRACT_RECORD_FIELDS:
            raise ValueError(f"MSP duplicate exclusion record fields mismatch at index {index}")
        identifier = str(record.get("utterance_id", ""))
        identifiers.append(identifier)
        if identifier not in group_by_member:
            raise ValueError(f"MSP duplicate exclusion target is not an audit candidate: {identifier}")
        audit_record = audit_by_id[identifier]
        expected_record = {
            "utterance_id": identifier,
            "audio_relpath": audit_record["audio_relpath"],
            "source_split": audit_record["source_split"],
            "split": audit_record["split"],
            "speaker_id": audit_record["speaker_id"],
            "original_emotion": audit_record["original_emotion"],
            "mapped_emotion": audit_record["mapped_emotion"],
            "byte_sha256": audit_record["byte_sha256"],
            "decoded_waveform_sha256": audit_record["decoded_waveform_sha256"],
            "duplicate_group_id": group_by_member[identifier]["group_id"],
            "exclusion_reason": MSP_DUPLICATE_EXCLUSION_REASON,
        }
        if record != expected_record:
            raise ValueError(f"MSP duplicate exclusion record does not match audit: {identifier}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("MSP duplicate exclusion contract has duplicate utterance IDs")

    approved = set(identifiers)
    unresolved = []
    for group in audit_payload["duplicate_groups"]:
        remaining_splits = {
            str(audit_by_id[str(identifier)]["split"])
            for identifier in group["member_ids"]
            if str(identifier) not in approved
        }
        if bool(group["cross_split"]) and len(remaining_splits) > 1:
            unresolved.append(str(group["group_id"]))
    if unresolved:
        raise ValueError(f"unresolved cross-split duplicate groups remain: {unresolved[:5]}")

    remaining = [record for record in audit_payload["records"] if record["utterance_id"] not in approved]
    expected_counts = {
        "audited_available": len(audit_payload["records"]),
        "approved_duplicate_exclusions": len(records),
        "final_included": len(remaining),
        "split_counts": dict(sorted(Counter(str(record["split"]) for record in remaining).items())),
    }
    if payload.get("post_exclusion_counts") != expected_counts:
        raise ValueError("MSP duplicate exclusion contract post-exclusion counts mismatch")

    actual_sha256 = normalized_duplicate_exclusion_contract_sha256(payload)
    stored_sha256 = _require_sha256(
        payload.get("normalized_sha256"),
        "MSP duplicate exclusion contract normalized SHA-256",
    )
    if stored_sha256 != actual_sha256:
        raise ValueError("MSP duplicate exclusion contract normalized SHA-256 mismatch")
    if expected_sha256 is not None:
        approved_sha256 = _require_sha256(expected_sha256, "approved duplicate exclusion SHA-256")
        if approved_sha256 != actual_sha256:
            raise ValueError("approved MSP duplicate exclusion SHA-256 mismatch")
    return {
        "schema_version": MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION,
        "normalized_sha256": actual_sha256,
        "audit_normalized_sha256": audit_report["normalized_sha256"],
        "count": len(records),
        "post_exclusion_counts": dict(expected_counts),
    }


def load_msp_audio_duplicate_exclusion_contract(
    path: str | Path,
    audit_payload: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate an msp_audio_duplicate_exclusions_v1 JSON file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MSP duplicate exclusion contract JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MSP duplicate exclusion contract must be a JSON object")
    report = validate_msp_audio_duplicate_exclusion_contract(
        payload,
        audit_payload,
        expected_sha256=expected_sha256,
    )
    return payload, report


def write_msp_audio_duplicate_exclusion_contract(
    payload: Mapping[str, Any],
    audit_payload: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Atomically write a validated duplicate-exclusion contract."""
    validate_msp_audio_duplicate_exclusion_contract(payload, audit_payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(output)
    return output


def generate_msp_audio_duplicate_exclusion_contract(
    audit_path: str | Path,
    approved_utterance_ids: Sequence[str],
    output: str | Path,
) -> dict[str, Any]:
    """Generate and persist a duplicate-exclusion contract from reviewed audit IDs."""
    audit_payload, audit_report = load_msp_audio_duplicate_audit(audit_path)
    payload = build_msp_audio_duplicate_exclusion_contract(audit_payload, approved_utterance_ids)
    destination = write_msp_audio_duplicate_exclusion_contract(payload, audit_payload, output)
    return {
        "dataset": "msp_podcast",
        "output": str(destination),
        "schema_version": payload["schema_version"],
        "normalized_sha256": payload["normalized_sha256"],
        "audit_normalized_sha256": audit_report["normalized_sha256"],
        "count": payload["count"],
        "post_exclusion_counts": payload["post_exclusion_counts"],
    }


def verify_msp_audio_duplicate_audit_freshness(
    payload: Mapping[str, Any],
    metadata_rows: Iterable[Mapping[str, Any]],
    audio_paths: Mapping[str, str | Path],
    *,
    missing_exclusion_contract_sha256: str,
) -> dict[str, Any]:
    """Re-hash current target files and reject a stale or differently scoped audit."""
    report = validate_msp_audio_duplicate_audit(payload)
    missing_sha256 = _require_sha256(
        missing_exclusion_contract_sha256,
        "missing-audio exclusion contract SHA-256",
    )
    if payload["missing_audio_exclusion_contract"]["normalized_sha256"] != missing_sha256:
        raise ValueError("MSP duplicate audit is stale: missing-audio exclusion contract SHA-256 changed")

    rows = list(metadata_rows)
    row_by_id = {str(row["utterance_id"]): row for row in rows}
    audit_by_id = {str(record["utterance_id"]): record for record in payload["records"]}
    if len(row_by_id) != len(rows) or set(row_by_id) != set(audit_by_id) or set(audio_paths) != set(audit_by_id):
        raise ValueError("MSP duplicate audit is stale: target utterance set changed")
    comparable = (
        "audio_relpath",
        "source_split",
        "split",
        "speaker_id",
        "original_emotion",
        "mapped_emotion",
    )
    for identifier, audit_record in audit_by_id.items():
        row = row_by_id[identifier]
        for field in comparable:
            if str(row.get(field)) != str(audit_record[field]):
                raise ValueError(f"MSP duplicate audit is stale: metadata changed for {identifier}/{field}")
        path = Path(audio_paths[identifier])
        if not path.is_file():
            raise ValueError(f"MSP duplicate audit is stale: audio is missing for {identifier}")
        current = inspect_audio(path, compute_sha256=True)
        expected = {
            "audio_sha256": audit_record["byte_sha256"],
            "audio_size_bytes": audit_record["audio_size_bytes"],
            "sample_rate_hz": audit_record["sample_rate_hz"],
            "channels": audit_record["channels"],
            "num_samples": audit_record["num_frames"],
        }
        for field, expected_value in expected.items():
            if current[field] != expected_value:
                raise ValueError(f"MSP duplicate audit is stale: current audio mismatch for {identifier}/{field}")
    return {
        "status": "ok",
        "audit_normalized_sha256": report["normalized_sha256"],
        "verified_audio": len(audit_by_id),
    }


def manifest_duplicate_provenance_signature(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate and summarize duplicate provenance embedded in MSP manifest rows."""
    rows = list(records)
    present = [row for row in rows if any(row.get(field) is not None for field in _MANIFEST_PROVENANCE_FIELDS)]
    duplicate_excluded = [row for row in rows if MSP_DUPLICATE_EXCLUSION_REASON in row.get("exclusion_reasons", [])]
    if not present:
        if duplicate_excluded:
            raise ValueError("manifest duplicate exclusions lack audit and contract provenance")
        return None
    if len(present) != len(rows):
        raise ValueError("manifest duplicate provenance must be present on every row")
    if {str(row.get("dataset")) for row in rows} != {"msp_podcast"}:
        raise ValueError("MSP duplicate provenance is valid only for an MSP manifest")
    for field in _MANIFEST_PROVENANCE_FIELDS:
        if len({row.get(field) for row in rows}) != 1:
            raise ValueError(f"manifest duplicate provenance is inconsistent for {field}")
    if {str(row["duplicate_audit_schema_version"]) for row in rows} != {MSP_DUPLICATE_AUDIT_SCHEMA_VERSION}:
        raise ValueError("manifest duplicate audit schema version mismatch")
    if {str(row["duplicate_exclusion_contract_schema_version"]) for row in rows} != {
        MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION
    }:
        raise ValueError("manifest duplicate exclusion contract schema version mismatch")
    audit_sha256 = _require_sha256(rows[0]["duplicate_audit_sha256"], "manifest duplicate audit SHA-256")
    contract_sha256 = _require_sha256(
        rows[0]["duplicate_exclusion_contract_sha256"],
        "manifest duplicate exclusion contract SHA-256",
    )
    target_count = int(rows[0]["duplicate_audit_target_count"])
    if target_count <= 0:
        raise ValueError("manifest duplicate audit target count must be positive")
    if any(bool(row.get("included")) for row in duplicate_excluded):
        raise ValueError("manifest approved duplicate exclusions must have included=false")
    for row in duplicate_excluded:
        if (
            row.get("speaker_id_status") != "known"
            or row.get("mapped_emotion") not in {"anger", "happy", "sadness", "disgust"}
            or row.get("split") not in {"train", "validation", "test"}
        ):
            raise ValueError("manifest duplicate exclusion target is outside the MSP research subset")
    pre_duplicate_eligible = [
        row
        for row in rows
        if bool(row.get("included"))
        or MSP_DUPLICATE_EXCLUSION_REASON in row.get("exclusion_reasons", [])
    ]
    if len(pre_duplicate_eligible) != target_count:
        raise ValueError("manifest duplicate audit target count mismatch")
    final_included = target_count - len(duplicate_excluded)
    if sum(bool(row.get("included")) for row in rows) != final_included:
        raise ValueError("manifest final included count does not match duplicate exclusion contract")
    split_counts = dict(sorted(Counter(str(row["split"]) for row in duplicate_excluded).items()))
    return {
        "audit": {
            "schema_version": MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
            "normalized_sha256": audit_sha256,
            "target_count": target_count,
        },
        "exclusion_contract": {
            "schema_version": MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION,
            "normalized_sha256": contract_sha256,
            "audit_normalized_sha256": audit_sha256,
            "count": len(duplicate_excluded),
            "excluded_split_counts": split_counts,
            "final_included": final_included,
        },
    }


__all__ = [
    "MSP_DUPLICATE_AUDIT_SCHEMA_VERSION",
    "MSP_DUPLICATE_EXCLUSION_REASON",
    "MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION",
    "build_msp_audio_duplicate_audit",
    "build_msp_audio_duplicate_exclusion_contract",
    "generate_msp_audio_duplicate_exclusion_contract",
    "load_msp_audio_duplicate_audit",
    "load_msp_audio_duplicate_exclusion_contract",
    "manifest_duplicate_provenance_signature",
    "normalized_duplicate_audit_sha256",
    "normalized_duplicate_exclusion_contract_sha256",
    "validate_msp_audio_duplicate_audit",
    "validate_msp_audio_duplicate_exclusion_contract",
    "verify_msp_audio_duplicate_audit_freshness",
    "write_msp_audio_duplicate_audit",
    "write_msp_audio_duplicate_candidates_csv",
    "write_msp_audio_duplicate_exclusion_contract",
]
