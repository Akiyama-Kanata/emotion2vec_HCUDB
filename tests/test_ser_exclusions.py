"""Validate the fixed MSP missing-audio contract without feature extraction or training."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ser_pipeline.contracts import MANIFEST_SCHEMA_VERSION
from ser_pipeline.cli import build_parser
from ser_pipeline.exclusions import (
    MSP_EXCLUSION_REASON,
    MSP_EXPECTED_ELIGIBLE_COUNT,
    MSP_EXPECTED_EXCLUDED_COUNT,
    MSP_EXPECTED_INCLUDED_COUNT,
    build_msp_missing_audio_exclusion_contract,
    normalized_exclusion_contract_sha256,
    reconcile_msp_exclusion_contract,
    validate_msp_missing_audio_exclusion_contract,
    write_msp_missing_audio_exclusion_contract,
)
from ser_pipeline.evaluation import evaluation_set_signature
from ser_pipeline.manifest import (
    build_manifest,
    generate_msp_missing_audio_exclusion_contract,
    load_manifest,
    validate_manifest_records,
)
from ser_pipeline.splits import MSP_SPLIT_VERSION
from ser_pipeline.study import DatasetArtifacts, bundle_msp_exclusion_contract


ORIGINAL_TO_MAPPED = {"A": ("anger", 0), "H": ("happy", 1), "S": ("sadness", 2), "D": ("disgust", 3)}
SOURCE_TO_SPLIT = {"Train": "train", "Development": "validation", "Test1": "test"}


def metadata_row(index: int, original: str, source_split: str) -> dict:
    mapped, class_index = ORIGINAL_TO_MAPPED[original]
    utterance_id = f"MSP-PODCAST_SYNTH_{index:05d}"
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "msp_podcast",
        "dataset_release": "R1.10",
        "utterance_id": utterance_id,
        "audio_relpath": f"Audio/{utterance_id}.wav",
        "audio_sha256": None,
        "speaker_id": f"speaker_{index:05d}",
        "speaker_id_status": "known",
        "group_id": f"speaker_{index:05d}",
        "session_id": f"podcast_{index:05d}",
        "source_split": source_split,
        "split": SOURCE_TO_SPLIT[source_split],
        "split_version": MSP_SPLIT_VERSION,
        "original_emotion": original,
        "mapped_emotion": mapped,
        "class_index": class_index,
        "mapping_version": "msp_podcast_r1_10_primary_v1",
        "included": True,
        "exclusion_reasons": [],
        "approximate_mapping": False,
        "audio_size_bytes": None,
        "sample_rate_hz": None,
        "channels": None,
        "num_samples": None,
        "duration_seconds": None,
        "source_metadata": {},
    }


def fixed_missing_rows() -> list[dict]:
    labels = ["A"] * 378 + ["H"] * 392 + ["S"] * 80 + ["D"] * 24
    splits = ["Train"] * 520 + ["Development"] * 210 + ["Test1"] * 144
    return [metadata_row(index, label, split) for index, (label, split) in enumerate(zip(labels, splits))]


def all_eligible_rows() -> list[dict]:
    rows = fixed_missing_rows()
    present_count = MSP_EXPECTED_ELIGIBLE_COUNT - len(rows)
    split_cycle = ("Train", "Development", "Test1")
    for offset in range(present_count):
        rows.append(metadata_row(len(rows), "A", split_cycle[offset % len(split_cycle)]))
    return rows


class MspExclusionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.missing_rows = fixed_missing_rows()
        cls.rows = all_eligible_rows()
        cls.payload = build_msp_missing_audio_exclusion_contract(cls.missing_rows)
        cls.contract_ids = {record["utterance_id"] for record in cls.payload["records"]}

    def test_fixed_contract_generation_is_deterministic(self):
        self.assertEqual(MSP_EXPECTED_EXCLUDED_COUNT, 874)
        self.assertEqual(MSP_EXPECTED_INCLUDED_COUNT, 25_111)
        self.assertEqual(self.payload["count"], 874)
        self.assertEqual(self.payload["expected_included_count"], 25_111)
        self.assertEqual(self.payload["counts"]["original_emotion"], {"A": 378, "D": 24, "H": 392, "S": 80})
        self.assertEqual(
            self.payload["counts"]["official_split"],
            {"Development": 210, "Test1": 144, "Train": 520},
        )
        filenames = [record["filename"] for record in self.payload["records"]]
        self.assertEqual(filenames, sorted(filenames, key=lambda value: (value.casefold(), value)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            with (
                patch("ser_pipeline.manifest.read_dataset", return_value=iter(self.missing_rows)),
                patch("ser_pipeline.manifest._audio_inventory", return_value=({}, 0)),
            ):
                first_report = generate_msp_missing_audio_exclusion_contract(root, first)
            with (
                patch("ser_pipeline.manifest.read_dataset", return_value=iter(self.missing_rows)),
                patch("ser_pipeline.manifest._audio_inventory", return_value=({}, 0)),
            ):
                second_report = generate_msp_missing_audio_exclusion_contract(root, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_report["normalized_sha256"], second_report["normalized_sha256"])

    def test_strict_manifest_succeeds_only_for_the_approved_missing_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "msp_missing_audio_exclusions_v1.json"
            manifest_path = root / "manifest.jsonl"
            write_msp_missing_audio_exclusion_contract(self.payload, contract_path)
            rows = [dict(row, exclusion_reasons=list(row["exclusion_reasons"])) for row in self.rows]
            available = {
                row["audio_relpath"]: root / row["audio_relpath"]
                for row in rows
                if row["utterance_id"] not in self.contract_ids
            }

            def fake_inspect_audio(path, *, compute_sha256=True):
                digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
                return {
                    "audio_sha256": digest if compute_sha256 else None,
                    "audio_size_bytes": 320,
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "num_samples": 160,
                    "duration_seconds": 0.01,
                }

            with (
                patch("ser_pipeline.manifest.read_dataset", return_value=iter(rows)),
                patch("ser_pipeline.manifest._audio_inventory", return_value=(available, 0)),
                patch("ser_pipeline.manifest.inspect_audio", side_effect=fake_inspect_audio),
            ):
                report = build_manifest(
                    "msp_podcast",
                    root,
                    manifest_path,
                    strict=True,
                    inspect_excluded_audio=False,
                    approved_exclusion_contract=contract_path,
                    expected_exclusion_sha256=self.payload["normalized_sha256"],
                )

            self.assertEqual(report["included"], 25_111)
            self.assertEqual(report["approved_missing_audio_exclusions"], 874)
            self.assertEqual(report["missing_included_audio"], 0)
            self.assertEqual(report["exclusion_contract_sha256"], self.payload["normalized_sha256"])
            manifest_rows = load_manifest(manifest_path)
            excluded = [row for row in manifest_rows if MSP_EXCLUSION_REASON in row["exclusion_reasons"]]
            self.assertEqual(len(excluded), 874)
            self.assertTrue(all(not row["included"] for row in excluded))
            validation = validate_manifest_records(manifest_rows)
            self.assertEqual(validation["included"], 25_111)
            self.assertEqual(
                validation["exclusion_contract"]["normalized_sha256"],
                self.payload["normalized_sha256"],
            )
            self.assertEqual(
                validation["splits"]["msp_podcast"]["split_counts"],
                report["validation"]["splits"]["msp_podcast"]["split_counts"],
            )
            evaluation_signature = evaluation_set_signature(manifest_path, "msp_podcast")
            self.assertEqual(
                evaluation_signature["exclusion_contract"]["normalized_sha256"],
                self.payload["normalized_sha256"],
            )
            self.assertEqual(evaluation_signature["manifest_sha256"], report["manifest_sha256"])
            cache_root = root / "cache"
            cache_root.mkdir()
            (cache_root / "cache_meta.json").write_text(
                json.dumps(
                    {
                        "cache_id": "synthetic-cache-id",
                        "exclusion_contract": validation["exclusion_contract"],
                    }
                ),
                encoding="utf-8",
            )
            bundled = bundle_msp_exclusion_contract(
                DatasetArtifacts(manifest_path, cache_root, contract_path),
                root / "formal-artifacts",
            )
            self.assertEqual(bundled["cache_id"], "synthetic-cache-id")
            self.assertEqual(bundled["manifest_sha256"], report["manifest_sha256"])
            self.assertEqual(bundled["normalized_sha256"], self.payload["normalized_sha256"])
            self.assertTrue(Path(bundled["path"]).is_file())

    def test_contract_and_current_inventory_rejections(self):
        missing_ids = set(self.contract_ids)

        recovered = set(missing_ids)
        recovered.remove(next(iter(recovered)))
        with self.assertRaisesRegex(ValueError, "stale"):
            reconcile_msp_exclusion_contract(self.rows, recovered, self.payload)

        outside = set(missing_ids)
        outside.add(next(row["utterance_id"] for row in self.rows if row["utterance_id"] not in missing_ids))
        with self.assertRaisesRegex(ValueError, "outside"):
            reconcile_msp_exclusion_contract(self.rows, outside, self.payload)

        mismatch_rows = list(self.rows)
        mismatch_rows[0] = dict(mismatch_rows[0], original_emotion="H", mapped_emotion="happy", class_index=1)
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            reconcile_msp_exclusion_contract(mismatch_rows, missing_ids, self.payload)

        duplicate = copy.deepcopy(self.payload)
        duplicate["records"][1] = dict(duplicate["records"][0])
        duplicate["normalized_sha256"] = normalized_exclusion_contract_sha256(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_msp_missing_audio_exclusion_contract(duplicate)

        wrong_count = copy.deepcopy(self.payload)
        wrong_count["count"] = 873
        wrong_count["normalized_sha256"] = normalized_exclusion_contract_sha256(wrong_count)
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            validate_msp_missing_audio_exclusion_contract(wrong_count)

        wrong_breakdown = copy.deepcopy(self.payload)
        wrong_breakdown["counts"]["official_split"]["Train"] -= 1
        wrong_breakdown["normalized_sha256"] = normalized_exclusion_contract_sha256(wrong_breakdown)
        with self.assertRaisesRegex(ValueError, "official split counts mismatch"):
            validate_msp_missing_audio_exclusion_contract(wrong_breakdown)

        wrong_sha = copy.deepcopy(self.payload)
        wrong_sha["normalized_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "normalized SHA-256 mismatch"):
            validate_msp_missing_audio_exclusion_contract(wrong_sha)
        with self.assertRaisesRegex(ValueError, "approved MSP exclusion SHA-256 mismatch"):
            validate_msp_missing_audio_exclusion_contract(self.payload, expected_sha256="0" * 64)

    def test_manifest_refuses_unset_approval_sha_before_reading_data(self):
        with self.assertRaisesRegex(ValueError, "requires an expected SHA-256"):
            build_manifest(
                "msp_podcast",
                Path("unused"),
                Path("unused.jsonl"),
                approved_exclusion_contract=Path("contract.json"),
            )

    def test_cli_exposes_generation_and_approval_options(self):
        parser = build_parser()
        generated = parser.parse_args(
            ["generate-msp-exclusion-contract", "--root", "dataset", "--output", "contract.json"]
        )
        self.assertEqual(generated.command, "generate-msp-exclusion-contract")
        built = parser.parse_args(
            [
                "build-manifest",
                "--dataset",
                "msp_podcast",
                "--root",
                "dataset",
                "--output",
                "manifest.jsonl",
                "--approved-exclusion-contract",
                "contract.json",
                "--expected-exclusion-sha256",
                "a" * 64,
            ]
        )
        self.assertEqual(built.approved_exclusion_contract, Path("contract.json"))
        self.assertEqual(built.expected_exclusion_sha256, "a" * 64)


if __name__ == "__main__":
    unittest.main()
