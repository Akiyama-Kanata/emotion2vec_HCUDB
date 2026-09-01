"""Test exact MSP audio duplicate audits with synthetic WAV files only."""

import copy
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ser_pipeline.cache import validate_cache
from ser_pipeline.cli import build_parser
from ser_pipeline.contracts import MANIFEST_SCHEMA_VERSION
from ser_pipeline.duplicates import (
    MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
    MSP_DUPLICATE_EXCLUSION_REASON,
    MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION,
    build_msp_audio_duplicate_audit,
    build_msp_audio_duplicate_exclusion_contract,
    load_msp_audio_duplicate_audit,
    load_msp_audio_duplicate_exclusion_contract,
    normalized_duplicate_audit_sha256,
    validate_msp_audio_duplicate_audit,
    validate_msp_audio_duplicate_exclusion_contract,
    verify_msp_audio_duplicate_audit_freshness,
    write_msp_audio_duplicate_audit,
    write_msp_audio_duplicate_candidates_csv,
    write_msp_audio_duplicate_exclusion_contract,
)
from ser_pipeline.evaluation import build_evaluation_result, evaluation_set_signature
from ser_pipeline.features import EncoderInfo, extract_feature_cache
from ser_pipeline.manifest import validate_manifest_records, write_manifest
from ser_pipeline.splits import MSP_SPLIT_VERSION
from ser_pipeline.study import DatasetArtifacts, bundle_msp_duplicate_provenance


MISSING_CONTRACT_SHA256 = "b" * 64
ORIGINAL_TO_MAPPED = {
    "A": ("anger", 0),
    "H": ("happy", 1),
    "S": ("sadness", 2),
    "D": ("disgust", 3),
}


class FakeEncoder:
    info = EncoderInfo("fake_duplicate_test_encoder", "e" * 64, 4)

    def extract(self, waveform):
        return np.repeat(np.asarray(waveform, dtype=np.float32)[:, None], 4, axis=1)


class MspAudioDuplicateAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio_root = self.root / "dataset"
        audio_dir = self.audio_root / "Audio"
        audio_dir.mkdir(parents=True)

        cross_waveform = np.asarray([0.0, 0.5, -0.5, 0.25, -0.25, 0.125, 0.0, -0.125], dtype=np.float32)
        sf.write(audio_dir / "cross_train.wav", cross_waveform, 16000, subtype="PCM_16")
        shutil.copyfile(audio_dir / "cross_train.wav", audio_dir / "cross_validation.wav")

        representation_waveform = np.asarray([0.0, 0.5, -0.5, 0.25, -0.25, 0.0], dtype=np.float32)
        sf.write(audio_dir / "representation_pcm16.wav", representation_waveform, 16000, subtype="PCM_16")
        sf.write(audio_dir / "representation_float.wav", representation_waveform, 16000, subtype="FLOAT")

        one_sample_left = np.asarray([0.0, 0.25, 0.5, -0.25, -0.5], dtype=np.float32)
        one_sample_right = one_sample_left.copy()
        one_sample_right[2] += np.float32(1 / 32768)
        sf.write(audio_dir / "one_sample_left.wav", one_sample_left, 16000, subtype="PCM_16")
        sf.write(audio_dir / "one_sample_right.wav", one_sample_right, 16000, subtype="PCM_16")

        specifications = (
            ("cross_train", "Train", "train", "A", "speaker_cross_train"),
            ("cross_validation", "Development", "validation", "A", "speaker_cross_validation"),
            ("representation_pcm16", "Development", "validation", "H", "speaker_representation_pcm"),
            ("representation_float", "Development", "validation", "S", "speaker_representation_float"),
            ("one_sample_left", "Train", "train", "D", "speaker_one_sample_left"),
            ("one_sample_right", "Test1", "test", "D", "speaker_one_sample_right"),
        )
        self.rows = []
        self.paths = {}
        for identifier, source_split, split, original, speaker in specifications:
            mapped, class_index = ORIGINAL_TO_MAPPED[original]
            relpath = f"Audio/{identifier}.wav"
            self.paths[identifier] = self.audio_root / relpath
            self.rows.append(
                {
                    "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                    "dataset": "msp_podcast",
                    "dataset_release": "R1.10",
                    "utterance_id": identifier,
                    "audio_relpath": relpath,
                    "audio_sha256": None,
                    "speaker_id": speaker,
                    "speaker_id_status": "known",
                    "group_id": speaker,
                    "session_id": "synthetic",
                    "source_split": source_split,
                    "split": split,
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
                }
            )
        self.audit = build_msp_audio_duplicate_audit(
            self.rows,
            self.paths,
            missing_exclusion_contract_schema_version="msp_missing_audio_exclusions_v1",
            missing_exclusion_contract_sha256=MISSING_CONTRACT_SHA256,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_two_stage_exact_audit_and_deterministic_artifacts(self):
        self.assertEqual(self.audit["schema_version"], MSP_DUPLICATE_AUDIT_SCHEMA_VERSION)
        self.assertEqual(
            self.audit["summary"],
            {
                "target_files": 6,
                "decoded_waveform_candidates": 6,
                "duplicate_groups": 2,
                "duplicate_members": 4,
                "byte_exact_groups": 1,
                "decoded_waveform_exact_groups": 2,
                "within_split_groups": 1,
                "cross_split_groups": 1,
                "speaker_mismatch_groups": 2,
                "label_mismatch_groups": 1,
            },
        )
        by_id = {record["utterance_id"]: record for record in self.audit["records"]}
        self.assertEqual(by_id["cross_train"]["byte_sha256"], by_id["cross_validation"]["byte_sha256"])
        self.assertNotEqual(
            by_id["representation_pcm16"]["byte_sha256"],
            by_id["representation_float"]["byte_sha256"],
        )
        self.assertEqual(
            by_id["representation_pcm16"]["decoded_waveform_sha256"],
            by_id["representation_float"]["decoded_waveform_sha256"],
        )
        self.assertNotEqual(
            by_id["one_sample_left"]["decoded_waveform_sha256"],
            by_id["one_sample_right"]["decoded_waveform_sha256"],
        )
        self.assertFalse(self.audit["method"]["resampling"])
        self.assertIsNone(self.audit["method"]["tolerance"])
        self.assertIsNone(self.audit["method"]["similarity_threshold"])

        second = build_msp_audio_duplicate_audit(
            list(reversed(self.rows)),
            self.paths,
            missing_exclusion_contract_schema_version="msp_missing_audio_exclusions_v1",
            missing_exclusion_contract_sha256=MISSING_CONTRACT_SHA256,
        )
        self.assertEqual(self.audit, second)
        first_json, second_json = self.root / "first.json", self.root / "second.json"
        first_csv, second_csv = self.root / "first.csv", self.root / "second.csv"
        write_msp_audio_duplicate_audit(self.audit, first_json)
        write_msp_audio_duplicate_audit(second, second_json)
        write_msp_audio_duplicate_candidates_csv(self.audit, first_csv)
        write_msp_audio_duplicate_candidates_csv(second, second_csv)
        self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
        self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())
        with first_csv.open("r", encoding="utf-8", newline="") as source:
            candidates = list(csv.DictReader(source))
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            set(candidates[0]),
            {
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
            },
        )

    def test_contract_rejects_unreviewed_ids_and_unresolved_cross_split_groups(self):
        with self.assertRaisesRegex(ValueError, "not an audit candidate"):
            build_msp_audio_duplicate_exclusion_contract(self.audit, ["not-registered"])
        with self.assertRaisesRegex(ValueError, "contain duplicates"):
            build_msp_audio_duplicate_exclusion_contract(self.audit, ["cross_train", "cross_train"])
        with self.assertRaisesRegex(ValueError, "unresolved cross-split"):
            build_msp_audio_duplicate_exclusion_contract(self.audit, [])

        contract = build_msp_audio_duplicate_exclusion_contract(self.audit, ["cross_validation"])
        self.assertEqual(contract["schema_version"], MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION)
        self.assertEqual(contract["count"], 1)
        self.assertEqual(contract["post_exclusion_counts"]["final_included"], 5)
        self.assertEqual(contract["records"][0]["exclusion_reason"], MSP_DUPLICATE_EXCLUSION_REASON)
        contract_path = self.root / "duplicate_exclusions.json"
        audit_path = self.root / "audit.json"
        write_msp_audio_duplicate_audit(self.audit, audit_path)
        write_msp_audio_duplicate_exclusion_contract(contract, self.audit, contract_path)
        loaded_audit, _ = load_msp_audio_duplicate_audit(audit_path)
        _, report = load_msp_audio_duplicate_exclusion_contract(
            contract_path,
            loaded_audit,
            expected_sha256=contract["normalized_sha256"],
        )
        self.assertEqual(report["count"], 1)
        with self.assertRaisesRegex(ValueError, "approved MSP duplicate exclusion SHA-256 mismatch"):
            validate_msp_audio_duplicate_exclusion_contract(contract, self.audit, expected_sha256="0" * 64)

    def test_modified_audit_and_changed_audio_are_rejected(self):
        tampered = copy.deepcopy(self.audit)
        tampered["records"][0]["speaker_id"] = "changed-speaker"
        with self.assertRaisesRegex(ValueError, "groups|summary|SHA-256"):
            validate_msp_audio_duplicate_audit(tampered)

        verify_msp_audio_duplicate_audit_freshness(
            self.audit,
            self.rows,
            self.paths,
            missing_exclusion_contract_sha256=MISSING_CONTRACT_SHA256,
        )
        changed = np.asarray([0.0, 0.25, 0.75, -0.25, -0.5], dtype=np.float32)
        sf.write(self.paths["one_sample_right"], changed, 16000, subtype="PCM_16")
        with self.assertRaisesRegex(ValueError, "audit is stale"):
            verify_msp_audio_duplicate_audit_freshness(
                self.audit,
                self.rows,
                self.paths,
                missing_exclusion_contract_sha256=MISSING_CONTRACT_SHA256,
            )

        rehashed_tamper = copy.deepcopy(self.audit)
        rehashed_tamper["records"][0]["speaker_id"] = "changed-speaker"
        rehashed_tamper["duplicate_groups"] = []
        rehashed_tamper["summary"] = dict(rehashed_tamper["summary"], duplicate_groups=0)
        rehashed_tamper["normalized_sha256"] = normalized_duplicate_audit_sha256(rehashed_tamper)
        with self.assertRaises(ValueError):
            validate_msp_audio_duplicate_audit(rehashed_tamper)

    def test_manifest_cache_evaluation_and_limitations_share_duplicate_contract(self):
        contract = build_msp_audio_duplicate_exclusion_contract(self.audit, ["cross_validation"])
        approved = {record["utterance_id"] for record in contract["records"]}
        manifest_rows = []
        for audit_record in self.audit["records"]:
            source = next(row for row in self.rows if row["utterance_id"] == audit_record["utterance_id"])
            row = dict(source)
            row.update(
                {
                    "audio_sha256": audit_record["byte_sha256"],
                    "audio_size_bytes": audit_record["audio_size_bytes"],
                    "sample_rate_hz": audit_record["sample_rate_hz"],
                    "channels": audit_record["channels"],
                    "num_samples": audit_record["num_frames"],
                    "duration_seconds": audit_record["num_frames"] / audit_record["sample_rate_hz"],
                    "included": audit_record["utterance_id"] not in approved,
                    "exclusion_reasons": (
                        [MSP_DUPLICATE_EXCLUSION_REASON] if audit_record["utterance_id"] in approved else []
                    ),
                    "duplicate_audit_schema_version": MSP_DUPLICATE_AUDIT_SCHEMA_VERSION,
                    "duplicate_audit_sha256": self.audit["normalized_sha256"],
                    "duplicate_audit_target_count": len(self.audit["records"]),
                    "duplicate_exclusion_contract_schema_version": MSP_DUPLICATE_EXCLUSION_SCHEMA_VERSION,
                    "duplicate_exclusion_contract_sha256": contract["normalized_sha256"],
                }
            )
            manifest_rows.append(row)
        manifest_path = self.root / "manifest.jsonl"
        write_manifest(manifest_rows, manifest_path)
        validation = validate_manifest_records(manifest_rows)
        self.assertEqual(validation["included"], 5)
        self.assertEqual(validation["duplicate_audit"]["normalized_sha256"], self.audit["normalized_sha256"])
        self.assertEqual(
            validation["duplicate_exclusion_contract"]["normalized_sha256"],
            contract["normalized_sha256"],
        )

        cache_root = self.root / "cache"
        extraction = extract_feature_cache(
            manifest_path,
            self.audio_root,
            cache_root,
            FakeEncoder(),
            max_shard_frames=32,
        )
        cache_validation = validate_cache(cache_root, manifest_path)
        self.assertEqual(extraction["utterances"], 5)
        self.assertEqual(cache_validation["duplicate_audit"], validation["duplicate_audit"])
        self.assertEqual(
            cache_validation["duplicate_exclusion_contract"],
            validation["duplicate_exclusion_contract"],
        )
        audit_path = self.root / "audit.json"
        contract_path = self.root / "contract.json"
        write_msp_audio_duplicate_audit(self.audit, audit_path)
        write_msp_audio_duplicate_exclusion_contract(contract, self.audit, contract_path)
        bundled = bundle_msp_duplicate_provenance(
            DatasetArtifacts(
                manifest_path=manifest_path,
                cache_root=cache_root,
                duplicate_audit_path=audit_path,
                duplicate_exclusion_contract_path=contract_path,
            ),
            self.root / "study",
        )
        self.assertEqual(bundled["audit"]["normalized_sha256"], self.audit["normalized_sha256"])
        self.assertEqual(
            bundled["exclusion_contract"]["normalized_sha256"],
            contract["normalized_sha256"],
        )

        set_signature = evaluation_set_signature(manifest_path, "msp_podcast", "test")
        self.assertEqual(set_signature["duplicate_audit"], validation["duplicate_audit"])
        self.assertEqual(
            set_signature["duplicate_exclusion_contract"],
            validation["duplicate_exclusion_contract"],
        )
        result = build_evaluation_result(
            ["one_sample_right"],
            [3],
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            dataset="msp_podcast",
            split="test",
            set_signature=set_signature,
        )
        limitation = next(
            item for item in result["limitations"] if item["id"] == "msp_podcast_r1_10_approved_contract_subset_v1"
        )
        self.assertEqual(limitation["excluded_duplicate_utterances"], 1)
        self.assertEqual(limitation["included_utterances"], 5)

    def test_cli_exposes_separate_audit_decision_and_manifest_inputs(self):
        parser = build_parser()
        audited = parser.parse_args(
            [
                "audit-msp-audio-duplicates",
                "--root",
                "dataset",
                "--audit-output",
                "audit.json",
                "--candidates-csv-output",
                "candidates.csv",
                "--approved-missing-exclusion-contract",
                "missing.json",
                "--expected-missing-exclusion-sha256",
                "a" * 64,
            ]
        )
        self.assertEqual(audited.audit_output, Path("audit.json"))
        decided = parser.parse_args(
            [
                "generate-msp-duplicate-exclusion-contract",
                "--audit",
                "audit.json",
                "--approved-id",
                "one",
                "--approved-id",
                "two",
                "--output",
                "contract.json",
            ]
        )
        self.assertEqual(decided.approved_id, ["one", "two"])
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
                "missing.json",
                "--expected-exclusion-sha256",
                "a" * 64,
                "--duplicate-audit",
                "audit.json",
                "--approved-duplicate-exclusion-contract",
                "contract.json",
                "--expected-duplicate-exclusion-sha256",
                "b" * 64,
            ]
        )
        self.assertEqual(built.duplicate_audit, Path("audit.json"))
        self.assertEqual(built.approved_duplicate_exclusion_contract, Path("contract.json"))
        self.assertEqual(built.expected_duplicate_exclusion_sha256, "b" * 64)


if __name__ == "__main__":
    unittest.main()
