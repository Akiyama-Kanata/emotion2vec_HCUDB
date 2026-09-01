"""Build, audit, hash, and validate UTF-8 JSONL manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

from .audio import inspect_audio
from .contracts import LABEL_ORDER, MANIFEST_FIELDS, MANIFEST_SCHEMA_VERSION, dataset_contract, map_emotion
from .duplicates import (
    MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
    MSP_DUPLICATE_EXCLUSION_REASON,
    MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION,
    build_msp_audio_duplicate_audit,
    load_msp_audio_duplicate_audit,
    load_msp_audio_duplicate_exclusion_contract,
    manifest_duplicate_provenance_signature,
    verify_msp_audio_duplicate_audit_freshness,
    write_msp_audio_duplicate_audit,
    write_msp_audio_duplicate_candidates_csv,
)
from .exclusions import (
    MSP_EXCLUSION_REASON,
    MSP_EXCLUSION_SCHEMA_VERSION,
    build_msp_missing_audio_exclusion_contract,
    load_msp_missing_audio_exclusion_contract,
    manifest_exclusion_contract_signature,
    reconcile_msp_exclusion_contract,
    write_msp_missing_audio_exclusion_contract,
)
from .readers import read_dataset, resolved_dataset_root
from .splits import validate_split_integrity


def canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def manifest_sha256(path: str | Path) -> str:
    return records_sha256(load_manifest(path))


def utterance_id_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    ids = sorted(str(row["utterance_id"]) for row in records if bool(row.get("included")))
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def write_manifest(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as destination:
        for row in records:
            destination.write(canonical_json(row) + "\n")
    partial.replace(output)
    return output


def _audio_path(dataset_root: Path, relpath: str) -> Path:
    return dataset_root.joinpath(*Path(relpath).parts)


def _audio_inventory(dataset_root: Path) -> tuple[dict[str, Path], int, dict[str, Path]]:
    files: dict[str, Path] = {}
    appledouble = 0
    zero_byte: dict[str, Path] = {}
    for directory, _subdirs, names in os.walk(dataset_root):
        for name in names:
            if not name.lower().endswith(".wav"):
                continue
            if name.startswith("._"):
                appledouble += 1
                continue
            path = Path(directory) / name
            relative = path.relative_to(dataset_root).as_posix()
            if path.stat().st_size == 0:
                zero_byte[relative] = path
                continue
            files[relative] = path
    return files, appledouble, zero_byte


def _audio_basename_index(files: Mapping[str, Path]) -> dict[str, list[tuple[str, Path]]]:
    index: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for relative, path in files.items():
        index[PurePosixPath(relative).name.casefold()].append((relative, path))
    return index


def _resolve_inventory_audio(
    dataset: str,
    expected_relpath: str,
    files: Mapping[str, Path],
    basename_index: Mapping[str, list[tuple[str, Path]]],
) -> tuple[str, Path] | None:
    exact = files.get(expected_relpath)
    if exact is not None:
        return expected_relpath, exact
    if dataset != "msp_podcast":
        return None
    filename = PurePosixPath(expected_relpath).name.casefold()
    matches = basename_index.get(filename, [])
    if len(matches) > 1:
        examples = [relative for relative, _path in matches[:5]]
        raise ValueError(f"duplicate MSP WAV basename {filename!r}: {examples}")
    return matches[0] if matches else None


def generate_msp_missing_audio_exclusion_contract(
    root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Generate the fixed v1 contract from eligible MSP rows missing at this moment."""
    dataset_root = resolved_dataset_root("msp_podcast", root)
    rows = list(read_dataset("msp_podcast", dataset_root))
    ids = [str(row["utterance_id"]) for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate metadata utterance_id: {duplicates[:5]}")
    available_audio, _appledouble, _zero_byte = _audio_inventory(dataset_root)
    basename_index = _audio_basename_index(available_audio)
    missing_rows = [
        row
        for row in rows
        if bool(row["included"])
        and _resolve_inventory_audio(
            "msp_podcast",
            str(row["audio_relpath"]),
            available_audio,
            basename_index,
        )
        is None
    ]
    payload = build_msp_missing_audio_exclusion_contract(missing_rows)
    destination = write_msp_missing_audio_exclusion_contract(payload, output)
    return {
        "dataset": "msp_podcast",
        "output": str(destination),
        "schema_version": payload["schema_version"],
        "normalized_sha256": payload["normalized_sha256"],
        "count": payload["count"],
        "expected_included_count": payload["expected_included_count"],
        "counts": payload["counts"],
    }


def _msp_duplicate_audit_targets(
    rows: Iterable[Mapping[str, Any]],
    resolved_audio: Mapping[str, tuple[str, Path] | None],
    missing_contract_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    target_rows: list[dict[str, Any]] = []
    target_paths: dict[str, Path] = {}
    for source_row in rows:
        identifier = str(source_row["utterance_id"])
        if not bool(source_row["included"]) or identifier in missing_contract_ids:
            continue
        resolved = resolved_audio.get(identifier)
        if resolved is None:
            raise ValueError(f"MSP duplicate audit target audio is missing: {identifier}")
        resolved_relpath, path = resolved
        row = dict(source_row)
        row["audio_relpath"] = resolved_relpath
        target_rows.append(row)
        target_paths[identifier] = path
    return target_rows, target_paths


def generate_msp_audio_duplicate_audit(
    root: str | Path,
    audit_output: str | Path,
    candidates_csv_output: str | Path,
    *,
    approved_missing_audio_exclusion_contract: str | Path,
    expected_missing_audio_exclusion_sha256: str,
) -> dict[str, Any]:
    """Audit exact duplicates in the approved available MSP research subset."""
    dataset_root = resolved_dataset_root("msp_podcast", root)
    rows = list(read_dataset("msp_podcast", dataset_root))
    identifiers = [str(row["utterance_id"]) for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate metadata utterance_id: {duplicates[:5]}")
    available_audio, _appledouble, _zero_byte = _audio_inventory(dataset_root)
    basename_index = _audio_basename_index(available_audio)
    resolved_audio = {
        str(row["utterance_id"]): _resolve_inventory_audio(
            "msp_podcast",
            str(row["audio_relpath"]),
            available_audio,
            basename_index,
        )
        for row in rows
    }
    missing_payload, missing_report = load_msp_missing_audio_exclusion_contract(
        approved_missing_audio_exclusion_contract,
        expected_sha256=expected_missing_audio_exclusion_sha256,
    )
    missing_ids = {str(record["utterance_id"]) for record in missing_payload["records"]}
    currently_missing = {
        str(row["utterance_id"])
        for row in rows
        if bool(row["included"]) and resolved_audio[str(row["utterance_id"])] is None
    }
    reconcile_msp_exclusion_contract(rows, currently_missing, missing_payload)
    target_rows, target_paths = _msp_duplicate_audit_targets(rows, resolved_audio, missing_ids)
    payload = build_msp_audio_duplicate_audit(
        target_rows,
        target_paths,
        missing_exclusion_contract_schema_version=missing_payload["schema_version"],
        missing_exclusion_contract_sha256=missing_report["normalized_sha256"],
    )
    audit_path = write_msp_audio_duplicate_audit(payload, audit_output)
    csv_path = write_msp_audio_duplicate_candidates_csv(payload, candidates_csv_output)
    return {
        "dataset": "msp_podcast",
        "audit_output": str(audit_path),
        "candidates_csv_output": str(csv_path),
        "schema_version": payload["schema_version"],
        "normalized_sha256": payload["normalized_sha256"],
        "missing_audio_exclusion_contract_sha256": missing_report["normalized_sha256"],
        "summary": payload["summary"],
    }


def build_manifest(
    dataset: str,
    root: str | Path,
    output: str | Path,
    *,
    strict: bool = True,
    inspect_excluded_audio: bool = True,
    approved_exclusion_contract: str | Path | None = None,
    expected_exclusion_sha256: str | None = None,
    duplicate_audit: str | Path | None = None,
    approved_duplicate_exclusion_contract: str | Path | None = None,
    expected_duplicate_exclusion_sha256: str | None = None,
) -> dict[str, Any]:
    normalized = str(dataset).strip().lower()
    if approved_exclusion_contract is not None and normalized != "msp_podcast":
        raise ValueError("approved missing-audio exclusions are supported only for MSP-Podcast")
    if expected_exclusion_sha256 is not None and approved_exclusion_contract is None:
        raise ValueError("expected exclusion SHA-256 requires an approved exclusion contract")
    if approved_exclusion_contract is not None and expected_exclusion_sha256 is None:
        raise ValueError("approved MSP exclusion contract requires an expected SHA-256")
    if approved_exclusion_contract is not None and not strict:
        raise ValueError("approved MSP exclusion contract requires strict=True")
    duplicate_inputs = (
        duplicate_audit,
        approved_duplicate_exclusion_contract,
        expected_duplicate_exclusion_sha256,
    )
    if any(value is not None for value in duplicate_inputs) and not all(value is not None for value in duplicate_inputs):
        raise ValueError("MSP duplicate audit, exclusion contract, and approval SHA-256 are required together")
    if any(value is not None for value in duplicate_inputs) and normalized != "msp_podcast":
        raise ValueError("approved duplicate exclusions are supported only for MSP-Podcast")
    if any(value is not None for value in duplicate_inputs) and approved_exclusion_contract is None:
        raise ValueError("MSP duplicate provenance requires the approved missing-audio exclusion contract")
    if any(value is not None for value in duplicate_inputs) and not strict:
        raise ValueError("approved MSP duplicate exclusions require strict=True")
    if approved_exclusion_contract is not None and not all(value is not None for value in duplicate_inputs):
        raise ValueError("strict MSP missing-audio exclusions require duplicate audit and approval inputs")
    dataset_root = resolved_dataset_root(normalized, root)
    rows = list(read_dataset(normalized, dataset_root))
    available_audio, _appledouble, _zero_byte = _audio_inventory(dataset_root)
    basename_index = _audio_basename_index(available_audio)
    ids = [row["utterance_id"] for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate metadata utterance_id: {duplicates[:5]}")

    resolved_audio: dict[str, tuple[str, Path] | None] = {}
    for row in rows:
        resolved_audio[str(row["utterance_id"])] = _resolve_inventory_audio(
            normalized,
            str(row["audio_relpath"]),
            available_audio,
            basename_index,
        )

    contract_payload: dict[str, Any] | None = None
    contract_report: dict[str, Any] | None = None
    contract_ids: set[str] = set()
    if approved_exclusion_contract is not None:
        contract_payload, contract_report = load_msp_missing_audio_exclusion_contract(
            approved_exclusion_contract,
            expected_sha256=expected_exclusion_sha256,
        )
        missing_eligible_ids = {
            str(row["utterance_id"])
            for row in rows
            if bool(row["included"]) and resolved_audio[str(row["utterance_id"])] is None
        }
        reconciliation = reconcile_msp_exclusion_contract(rows, missing_eligible_ids, contract_payload)
        contract_report = {**contract_report, **reconciliation, "path": str(Path(approved_exclusion_contract))}
        contract_ids = {str(record["utterance_id"]) for record in contract_payload["records"]}
        for row in rows:
            row["exclusion_contract_schema_version"] = MSP_EXCLUSION_SCHEMA_VERSION
            row["exclusion_contract_sha256"] = contract_report["normalized_sha256"]

    duplicate_audit_payload: dict[str, Any] | None = None
    duplicate_audit_report: dict[str, Any] | None = None
    duplicate_contract_payload: dict[str, Any] | None = None
    duplicate_contract_report: dict[str, Any] | None = None
    duplicate_contract_ids: set[str] = set()
    if duplicate_audit is not None:
        assert contract_payload is not None and contract_report is not None
        duplicate_audit_payload, duplicate_audit_report = load_msp_audio_duplicate_audit(duplicate_audit)
        missing_reference = duplicate_audit_payload["missing_audio_exclusion_contract"]
        if missing_reference != {
            "schema_version": MSP_EXCLUSION_SCHEMA_VERSION,
            "normalized_sha256": contract_report["normalized_sha256"],
        }:
            raise ValueError("MSP duplicate audit missing-audio exclusion contract provenance mismatch")
        target_rows, target_paths = _msp_duplicate_audit_targets(rows, resolved_audio, contract_ids)
        freshness = verify_msp_audio_duplicate_audit_freshness(
            duplicate_audit_payload,
            target_rows,
            target_paths,
            missing_exclusion_contract_sha256=contract_report["normalized_sha256"],
        )
        duplicate_audit_report = {**duplicate_audit_report, **freshness, "path": str(Path(duplicate_audit))}
        duplicate_contract_payload, duplicate_contract_report = load_msp_audio_duplicate_exclusion_contract(
            approved_duplicate_exclusion_contract,
            duplicate_audit_payload,
            expected_sha256=expected_duplicate_exclusion_sha256,
        )
        duplicate_contract_report = {
            **duplicate_contract_report,
            "path": str(Path(approved_duplicate_exclusion_contract)),
        }
        duplicate_contract_ids = {
            str(record["utterance_id"]) for record in duplicate_contract_payload["records"]
        }
        if duplicate_contract_ids & contract_ids:
            raise ValueError("MSP duplicate exclusions overlap missing-audio exclusions")
        if duplicate_audit_report["target_count"] != contract_report["expected_included_count"]:
            raise ValueError("MSP duplicate audit target count does not equal the missing-audio contract count")
        for row in rows:
            row["duplicate_audit_schema_version"] = MSP_DUPLICATE_AUDIT_SCHEMA_VERSION
            row["duplicate_audit_sha256"] = duplicate_audit_report["normalized_sha256"]
            row["duplicate_audit_target_count"] = duplicate_audit_report["target_count"]
            row["duplicate_exclusion_contract_schema_version"] = MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION
            row["duplicate_exclusion_contract_sha256"] = duplicate_contract_report["normalized_sha256"]

    missing_included: list[str] = []
    for row in rows:
        identifier = str(row["utterance_id"])
        resolved = resolved_audio[identifier]
        if identifier in contract_ids:
            row["included"] = False
            if MSP_EXCLUSION_REASON not in row["exclusion_reasons"]:
                row["exclusion_reasons"].append(MSP_EXCLUSION_REASON)
        if identifier in duplicate_contract_ids:
            row["included"] = False
            if MSP_DUPLICATE_EXCLUSION_REASON not in row["exclusion_reasons"]:
                row["exclusion_reasons"].append(MSP_DUPLICATE_EXCLUSION_REASON)
        eligible = bool(row["included"])
        if resolved is None:
            if eligible:
                missing_included.append(identifier)
                if not strict:
                    row["included"] = False
                    row["exclusion_reasons"].append("audio_missing")
            continue
        resolved_relpath, path = resolved
        row["audio_relpath"] = resolved_relpath
        if eligible or inspect_excluded_audio:
            metadata = inspect_audio(path, compute_sha256=eligible)
            row.update(metadata)
            if eligible and metadata["channels"] != 1:
                raise ValueError(f"included audio must be mono: {row['utterance_id']}")
    if missing_included and strict:
        raise ValueError(
            f"strict manifest requires every included audio file; missing {len(missing_included)} "
            f"(first: {missing_included[0]})"
        )
    validation = validate_manifest_records(rows, validate_splits=not missing_included)
    write_manifest(rows, output)
    report = {
        "dataset": normalized,
        "output": str(Path(output)),
        "total": len(rows),
        "included": sum(bool(row["included"]) for row in rows),
        "missing_included_audio": len(missing_included),
        "approved_missing_audio_exclusions": len(contract_ids),
        "approved_duplicate_exclusions": len(duplicate_contract_ids),
        "manifest_sha256": records_sha256(rows),
        "validation": validation,
    }
    if contract_report is not None:
        report["exclusion_contract_sha256"] = contract_report["normalized_sha256"]
        report["exclusion_contract"] = contract_report
    if duplicate_audit_report is not None and duplicate_contract_report is not None:
        report["duplicate_audit_sha256"] = duplicate_audit_report["normalized_sha256"]
        report["duplicate_exclusion_contract_sha256"] = duplicate_contract_report["normalized_sha256"]
        report["duplicate_audit"] = duplicate_audit_report
        report["duplicate_exclusion_contract"] = duplicate_contract_report
    return report


def audit_dataset(dataset: str, root: str | Path) -> dict[str, Any]:
    normalized = str(dataset).strip().lower()
    dataset_root = resolved_dataset_root(normalized, root)
    available_audio, appledouble, zero_byte_audio = _audio_inventory(dataset_root)
    basename_index = _audio_basename_index(available_audio)
    zero_byte_basename_index = _audio_basename_index(zero_byte_audio)
    total = 0
    eligible_count = 0
    missing_count = 0
    first_missing = None
    labels: Counter[str] = Counter()
    source_splits: Counter[str] = Counter()
    speakers: set[str] = set()
    test2 = 0
    matched_audio_relpaths: set[str] = set()
    metadata_missing_total = 0
    eligible_mapped_labels: Counter[str] = Counter()
    eligible_source_splits: Counter[str] = Counter()
    available_eligible_mapped_labels: Counter[str] = Counter()
    missing_eligible_original_labels: Counter[str] = Counter()
    missing_eligible_mapped_labels: Counter[str] = Counter()
    missing_eligible_source_splits: Counter[str] = Counter()
    missing_eligible_kinds: Counter[str] = Counter()
    missing_eligible_original_by_source_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_dataset(normalized, dataset_root):
        total += 1
        resolved = _resolve_inventory_audio(
            normalized,
            str(row["audio_relpath"]),
            available_audio,
            basename_index,
        )
        if resolved is None:
            metadata_missing_total += 1
        else:
            matched_audio_relpaths.add(resolved[0])
        labels[row["original_emotion"]] += 1
        source_splits[row["source_split"]] += 1
        if row["speaker_id_status"] == "known":
            speakers.add(row["speaker_id"])
        if row["source_split"] == "Test2":
            test2 += 1
        if row["included"]:
            eligible_count += 1
            mapped_label = str(row["mapped_emotion"])
            source_split = str(row["source_split"])
            original_emotion = str(row["original_emotion"])
            eligible_mapped_labels[mapped_label] += 1
            eligible_source_splits[source_split] += 1
            if resolved is None:
                zero_byte_resolved = _resolve_inventory_audio(
                    normalized,
                    str(row["audio_relpath"]),
                    zero_byte_audio,
                    zero_byte_basename_index,
                )
                missing_kind = "zero_byte" if zero_byte_resolved is not None else "absent"
                missing_count += 1
                missing_eligible_original_labels[original_emotion] += 1
                missing_eligible_mapped_labels[mapped_label] += 1
                missing_eligible_source_splits[source_split] += 1
                missing_eligible_kinds[missing_kind] += 1
                missing_eligible_original_by_source_split[source_split][original_emotion] += 1
                if first_missing is None:
                    first_missing = row["utterance_id"]
            else:
                available_eligible_mapped_labels[mapped_label] += 1
    return {
        "dataset": normalized,
        "total_metadata_rows": total,
        "eligible_primary_rows": eligible_count,
        "known_speakers": len(speakers),
        "missing_eligible_audio": missing_count,
        "first_missing_audio_id": first_missing,
        "eligible_mapped_label_counts": dict(sorted(eligible_mapped_labels.items())),
        "eligible_source_split_counts": dict(sorted(eligible_source_splits.items())),
        "available_eligible_mapped_label_counts": dict(sorted(available_eligible_mapped_labels.items())),
        "missing_eligible_original_label_counts": dict(sorted(missing_eligible_original_labels.items())),
        "missing_eligible_mapped_label_counts": dict(sorted(missing_eligible_mapped_labels.items())),
        "missing_eligible_source_split_counts": dict(sorted(missing_eligible_source_splits.items())),
        "missing_eligible_kind_counts": dict(sorted(missing_eligible_kinds.items())),
        "missing_eligible_original_by_source_split_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(missing_eligible_original_by_source_split.items())
        },
        "test2_audited_rows": test2,
        "appledouble_files_ignored": appledouble,
        "wav_files_found": len(available_audio) + len(zero_byte_audio),
        "zero_byte_audio_files_treated_as_missing": len(zero_byte_audio),
        "candidate_audio_files": len(available_audio),
        "valid_audio_files": len(available_audio),
        "metadata_audio_missing_total": metadata_missing_total,
        "unregistered_audio_files": len(set(available_audio) - matched_audio_relpaths),
        "original_label_counts": dict(sorted(labels.items())),
        "source_split_counts": dict(sorted(source_splits.items())),
    }


def validate_manifest_records(
    records: Iterable[Mapping[str, Any]],
    *,
    validate_splits: bool = True,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("manifest is empty")
    ids: set[str] = set()
    dataset_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(rows, 1):
        missing = [field for field in MANIFEST_FIELDS if field not in row]
        if missing:
            raise ValueError(f"manifest row {line_number} missing fields: {missing}")
        if row["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("manifest_schema_version mismatch")
        dataset = str(row["dataset"])
        contract = dataset_contract(dataset)
        if row["dataset_release"] != contract["dataset_release"]:
            raise ValueError(f"dataset_release mismatch: {row.get('utterance_id')}")
        decision = map_emotion(dataset, str(row["original_emotion"]))
        if row["mapping_version"] != decision.mapping_version:
            raise ValueError(f"mapping_version mismatch: {row.get('utterance_id')}")
        if row["mapped_emotion"] != decision.mapped_emotion or row["class_index"] != decision.class_index:
            raise ValueError(f"mapped label contract mismatch: {row.get('utterance_id')}")
        if bool(row["approximate_mapping"]) != decision.approximate_mapping:
            raise ValueError(f"approximate_mapping mismatch: {row.get('utterance_id')}")
        if row["speaker_id_status"] not in {"known", "unknown"}:
            raise ValueError(f"invalid speaker_id_status: {row.get('utterance_id')}")
        utterance_id = str(row["utterance_id"])
        if not utterance_id or utterance_id in ids:
            raise ValueError(f"duplicate or empty utterance_id: {utterance_id!r}")
        ids.add(utterance_id)
        relpath = str(row["audio_relpath"])
        if PurePosixPath(relpath).is_absolute() or PureWindowsPath(relpath).is_absolute():
            raise ValueError(f"audio_relpath must be relative: {utterance_id}")
        reasons = row["exclusion_reasons"]
        if not isinstance(reasons, list):
            raise ValueError(f"exclusion_reasons must be a list: {utterance_id}")
        if row["included"]:
            if reasons:
                raise ValueError(f"included row has exclusion reasons: {utterance_id}")
            if row["speaker_id_status"] == "unknown":
                raise ValueError(f"included Unknown speaker: {utterance_id}")
            if row["mapped_emotion"] not in LABEL_ORDER:
                raise ValueError(f"included row has invalid mapped emotion: {utterance_id}")
            expected_index = LABEL_ORDER.index(row["mapped_emotion"])
            if row["class_index"] != expected_index:
                raise ValueError(f"class_index mismatch: {utterance_id}")
            sha = row["audio_sha256"]
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError(f"included row requires audio_sha256: {utterance_id}")
            numeric = ("audio_size_bytes", "sample_rate_hz", "channels", "num_samples", "duration_seconds")
            if any(row[field] is None or float(row[field]) <= 0 for field in numeric):
                raise ValueError(f"included row has invalid audio metadata: {utterance_id}")
            if int(row["channels"]) != 1:
                raise ValueError(f"included audio must be mono: {utterance_id}")
        elif not reasons:
            raise ValueError(f"excluded row must preserve an exclusion reason: {utterance_id}")
        dataset_rows[dataset].append(row)
    exclusion_contracts: dict[str, Any] = {}
    duplicate_provenance: dict[str, Any] = {}
    for dataset, subset in dataset_rows.items():
        signature = manifest_exclusion_contract_signature(subset)
        if signature is not None:
            exclusion_contracts[dataset] = signature
        duplicate_signature = manifest_duplicate_provenance_signature(subset)
        if duplicate_signature is not None:
            duplicate_provenance[dataset] = duplicate_signature
    split_reports = {}
    if validate_splits:
        for dataset, subset in dataset_rows.items():
            if any(row["included"] for row in subset):
                split_reports[dataset] = validate_split_integrity(subset)
    return {
        "status": "ok",
        "total": len(rows),
        "included": sum(bool(row["included"]) for row in rows),
        "manifest_sha256": records_sha256(rows),
        "utterance_id_sha256": utterance_id_sha256(rows),
        "exclusion_contract": exclusion_contracts.get("msp_podcast"),
        "exclusion_contracts": exclusion_contracts,
        "duplicate_audit": duplicate_provenance.get("msp_podcast", {}).get("audit"),
        "duplicate_exclusion_contract": duplicate_provenance.get("msp_podcast", {}).get(
            "exclusion_contract"
        ),
        "duplicate_provenance": duplicate_provenance,
        "splits": split_reports,
    }


def validate_manifest_audio(records: Iterable[Mapping[str, Any]], root: str | Path) -> dict[str, Any]:
    rows = [row for row in records if row["included"]]
    if not rows:
        raise ValueError("manifest has no included rows to verify")
    datasets = {str(row["dataset"]) for row in rows}
    if len(datasets) != 1:
        raise ValueError("one audio root can verify only a single-dataset manifest")
    dataset = next(iter(datasets))
    dataset_root = resolved_dataset_root(dataset, root)
    for row in rows:
        path = _audio_path(dataset_root, str(row["audio_relpath"]))
        if not path.is_file():
            raise ValueError(f"included audio is missing: {row['utterance_id']}")
        actual = inspect_audio(path, compute_sha256=True)
        for field in (
            "audio_sha256",
            "audio_size_bytes",
            "sample_rate_hz",
            "channels",
            "num_samples",
        ):
            if actual[field] != row[field]:
                raise ValueError(f"included audio metadata mismatch for {field}: {row['utterance_id']}")
        if not np_isclose(float(actual["duration_seconds"]), float(row["duration_seconds"])):
            raise ValueError(f"included audio metadata mismatch for duration_seconds: {row['utterance_id']}")
    return {"status": "ok", "verified_audio": len(rows), "dataset": dataset}


def np_isclose(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def validate_manifest(path: str | Path, *, audio_root: str | Path | None = None) -> dict[str, Any]:
    rows = load_manifest(path)
    result = validate_manifest_records(rows)
    if audio_root is not None:
        result["audio"] = validate_manifest_audio(rows, audio_root)
    return result
