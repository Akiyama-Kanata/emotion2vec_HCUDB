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

    def _write_msp(self, include_missing=False):
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
                self._write_audio(self.root / "Audio" / filename)

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
        with self.assertRaisesRegex(ValueError, "strict manifest"):
            build_manifest("msp_podcast", self.root, self.root / "manifest.jsonl")

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
