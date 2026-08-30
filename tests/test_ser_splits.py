"""SER データの固定分割と話者・音声リーク検査を検証する。"""

import copy
import unittest

from ser_pipeline.splits import (
    IEMOCAP_SPLIT_VERSION,
    MSP_SPLIT_VERSION,
    derive_hcudb_split,
    hcudb_split_for_speaker,
    load_hcudb_split,
    msp_split,
    validate_split_integrity,
)


def row(dataset, utterance, speaker, split, audio_hash, source_split, split_version):
    return {
        "dataset": dataset,
        "utterance_id": utterance,
        "speaker_id": speaker,
        "speaker_id_status": "known",
        "split": split,
        "audio_sha256": audio_hash,
        "source_split": source_split,
        "split_version": split_version,
        "included": True,
    }


class SerSplitTest(unittest.TestCase):
    def test_fixed_hcudb_assignment(self):
        config = load_hcudb_split()
        self.assertEqual(config["split_version"], "hcudb1_speaker_split_v1")
        self.assertEqual(config["splits"]["train"], ["FA", "FB", "FD", "FH", "FI", "FL", "MC", "MJ", "MM", "MN"])
        self.assertEqual(config["splits"]["validation"], ["FF", "MK"])
        self.assertEqual(config["splits"]["test"], ["FG", "ME"])
        for split, speakers in config["splits"].items():
            for speaker in speakers:
                self.assertEqual(hcudb_split_for_speaker(speaker), split)
        all_speakers = [speaker for speakers in config["splits"].values() for speaker in speakers]
        self.assertEqual(
            derive_hcudb_split(all_speakers, seed=42),
            {name: sorted(speakers) for name, speakers in config["splits"].items()},
        )

    def test_msp_official_split_mapping_and_test2_exclusion(self):
        self.assertEqual(msp_split("Train"), "train")
        self.assertEqual(msp_split("Development"), "validation")
        self.assertEqual(msp_split("Test1"), "test")
        self.assertIsNone(msp_split("Test2"))
        with self.assertRaisesRegex(ValueError, "unknown MSP"):
            msp_split("random")

    def test_detects_speaker_utterance_audio_and_unknown_leakage(self):
        base = [
            row("msp_podcast", "a", "s1", "train", "a" * 64, "Train", MSP_SPLIT_VERSION),
            row("msp_podcast", "b", "s2", "validation", "b" * 64, "Development", MSP_SPLIT_VERSION),
            row("msp_podcast", "c", "s3", "test", "c" * 64, "Test1", MSP_SPLIT_VERSION),
        ]
        self.assertEqual(validate_split_integrity(base)["status"], "ok")
        for field, value, message in (
            ("speaker_id", "s1", "speaker leakage"),
            ("utterance_id", "a", "duplicate included utterance_id"),
            ("audio_sha256", "a" * 64, "audio hash leakage"),
        ):
            broken = copy.deepcopy(base)
            broken[1][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_split_integrity(broken)
        broken = copy.deepcopy(base)
        broken[0]["speaker_id_status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_split_integrity(broken)

    def test_iemocap_is_external_test_only(self):
        external = [row("iemocap", "a", "Ses01F", "test", "a" * 64, "all_sessions", IEMOCAP_SPLIT_VERSION)]
        self.assertEqual(validate_split_integrity(external)["split_counts"], {"test": 1})


if __name__ == "__main__":
    unittest.main()
