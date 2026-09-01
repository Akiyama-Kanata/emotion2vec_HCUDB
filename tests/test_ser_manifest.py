"""SER 用 JSONL マニフェストの生成、ハッシュ記録、監査を検証する。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ser_pipeline.manifest import (
    audit_dataset,
    build_manifest,
    load_manifest,
    manifest_sha256,
    records_sha256,
    validate_manifest,
)


class SerManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_audio(self, path, sample_rate=16000):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = (sum(path.name.encode("utf-8")) % 100 + 1) / 1000.0
        sf.write(path, np.full(sample_rate // 20, value, dtype=np.float32), sample_rate)

    def _write_msp(self, include_missing=False, nested_audio=False):
        labels = self.root / "Labels"
        labels.mkdir()
        rows = [
            ("train.wav", "A", "s1", "Train"),
            ("valid.wav", "H", "s2", "Development"),
            ("test.wav", "S", "s3", "Test1"),
            ("test2.wav", "D", "s4", "Test2"),
            ("unknown.wav", "D", "Unknown", "Train"),
            ("excluded.wav", "N", "s5", "Train"),
        ]
        with (labels / "labels_consensus.csv").open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(["FileName", "EmoClass", "EmoAct", "EmoVal", "EmoDom", "SpkrID", "Gender", "Split_Set"])
            for filename, emotion, speaker, split in rows:
                writer.writerow([filename, emotion, 1, 2, 3, speaker, "X", split])
        for filename, _emotion, _speaker, _split in rows:
            if filename != "excluded.wav" and not (include_missing and filename == "train.wav"):
                relative = Path("Audio") / "batch_001" / filename if nested_audio else Path("Audio") / filename
                self._write_audio(self.root / relative)

    def test_build_manifest_keeps_excluded_rows_and_is_stable(self):
        self._write_msp()
        first = self.root / "first.jsonl"
        second = self.root / "second.jsonl"
        result = build_manifest("msp_podcast", self.root, first, inspect_excluded_audio=False)
        build_manifest("msp_podcast", self.root, second, inspect_excluded_audio=False)
        rows = load_manifest(first)
        self.assertEqual(result["total"], 6)
        self.assertEqual(result["included"], 3)
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))
        self.assertEqual(records_sha256(rows), result["manifest_sha256"])
        by_id = {row["utterance_id"]: row for row in rows}
        self.assertEqual(by_id["test2"]["exclusion_reasons"], ["msp_test2_out_of_scope"])
        self.assertEqual(by_id["unknown"]["exclusion_reasons"], ["unknown_speaker"])
        self.assertEqual(by_id["excluded"]["exclusion_reasons"], ["label_not_in_primary_4"])
        self.assertFalse(Path(by_id["train"]["audio_relpath"]).is_absolute())
        self.assertEqual(validate_manifest(first)["status"], "ok")
        self.assertEqual(validate_manifest(first, audio_root=self.root)["audio"]["verified_audio"], 3)

    def test_strict_missing_audio_is_rejected_and_audit_reports_it(self):
        self._write_msp(include_missing=True)
        audit = audit_dataset("msp_podcast", self.root)
        self.assertEqual(audit["eligible_primary_rows"], 3)
        self.assertEqual(audit["missing_eligible_audio"], 1)
        self.assertEqual(audit["missing_eligible_original_label_counts"], {"A": 1})
        self.assertEqual(audit["missing_eligible_mapped_label_counts"], {"anger": 1})
        self.assertEqual(audit["missing_eligible_source_split_counts"], {"Train": 1})
        self.assertEqual(audit["missing_eligible_kind_counts"], {"absent": 1})
        self.assertEqual(audit["missing_eligible_original_by_source_split_counts"], {"Train": {"A": 1}})
        self.assertEqual(
            audit["available_eligible_mapped_label_counts"],
            {"happy": 1, "sadness": 1},
        )
        with self.assertRaisesRegex(ValueError, "strict manifest"):
            build_manifest("msp_podcast", self.root, self.root / "manifest.jsonl")

    def test_zero_byte_audio_is_treated_as_missing(self):
        self._write_msp()
        (self.root / "Audio" / "train.wav").write_bytes(b"")

        audit = audit_dataset("msp_podcast", self.root)

        self.assertEqual(audit["missing_eligible_audio"], 1)
        self.assertEqual(audit["first_missing_audio_id"], "train")
        self.assertEqual(audit["zero_byte_audio_files_treated_as_missing"], 1)
        self.assertEqual(audit["missing_eligible_kind_counts"], {"zero_byte": 1})
        self.assertEqual(audit["wav_files_found"], 5)
        self.assertEqual(audit["candidate_audio_files"], 4)
        with self.assertRaisesRegex(ValueError, "strict manifest"):
            build_manifest("msp_podcast", self.root, self.root / "manifest.jsonl", strict=True)

    def test_nested_msp_audio_is_resolved_by_unique_filename(self):
        self._write_msp(nested_audio=True)
        audit = audit_dataset("msp_podcast", self.root)
        self.assertEqual(audit["missing_eligible_audio"], 0)
        self.assertEqual(audit["unregistered_audio_files"], 0)

        manifest = self.root / "nested.jsonl"
        build_manifest("msp_podcast", self.root, manifest)
        rows = load_manifest(manifest)
        included_paths = {row["audio_relpath"] for row in rows if row["included"]}
        self.assertTrue(all(path.startswith("Audio/batch_001/") for path in included_paths))
        self.assertEqual(validate_manifest(manifest, audio_root=self.root)["audio"]["verified_audio"], 3)

    def test_nested_msp_duplicate_filename_is_rejected(self):
        self._write_msp(nested_audio=True)
        self._write_audio(self.root / "Audio" / "batch_002" / "train.wav")
        with self.assertRaisesRegex(ValueError, "duplicate MSP WAV basename"):
            audit_dataset("msp_podcast", self.root)

    def test_included_audio_decode_failure_is_not_auto_excluded(self):
        self._write_msp()
        (self.root / "Audio" / "train.wav").write_text("not a wave file", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unreadable audio"):
            build_manifest("msp_podcast", self.root, self.root / "manifest.jsonl", strict=True)
        self.assertFalse((self.root / "manifest.jsonl").exists())

    def test_split_validation_failure_does_not_write_manifest(self):
        self._write_msp()
        train = self.root / "Audio" / "train.wav"
        valid = self.root / "Audio" / "valid.wav"
        valid.write_bytes(train.read_bytes())
        manifest = self.root / "manifest.jsonl"

        with self.assertRaisesRegex(ValueError, "audio hash leakage"):
            build_manifest("msp_podcast", self.root, manifest, strict=True)

        self.assertFalse(manifest.exists())

    def test_invalid_json_and_absolute_paths_are_rejected(self):
        invalid = self.root / "invalid.jsonl"
        invalid.write_text("not json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSONL"):
            load_manifest(invalid)
        self._write_msp()
        manifest = self.root / "manifest.jsonl"
        build_manifest("msp_podcast", self.root, manifest, inspect_excluded_audio=False)
        rows = load_manifest(manifest)
        rows[0]["audio_relpath"] = str((self.root / "absolute.wav").resolve())
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "relative"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
