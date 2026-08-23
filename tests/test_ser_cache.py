import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ser_pipeline.audio import inspect_audio, load_audio_16k_mono
from ser_pipeline.cache import ShardedFeatureStore, validate_cache
from ser_pipeline.contracts import MANIFEST_SCHEMA_VERSION
from ser_pipeline.features import EncoderInfo, extract_feature_cache
from ser_pipeline.manifest import write_manifest
from ser_pipeline.preflight import disk_capacity_gate, estimate_full_extraction
from ser_pipeline.splits import MSP_SPLIT_VERSION


class FakeEncoder:
    info = EncoderInfo("fake_encoder", "f" * 64, 4)

    def extract(self, waveform):
        frames = max(1, len(waveform) // 400)
        base = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
        return np.concatenate([base + index for index in range(4)], axis=1).astype(np.float32)


class SerCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio_root = self.root / "audio_root"
        self.manifest = self.root / "manifest.jsonl"
        self.cache = self.root / "cache"
        rows = []
        specs = (("train", "Train"), ("validation", "Development"), ("test", "Test1"))
        for split_index, (split, source_split) in enumerate(specs):
            for item in range(2):
                utterance = f"{split}_{item}"
                relpath = f"Audio/{utterance}.wav"
                path = self.audio_root / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                waveform = np.full(1600 + 400 * item, 0.01 * (split_index + 1) + item / 1000, dtype=np.float32)
                sf.write(path, waveform, 16000)
                audio = inspect_audio(path)
                rows.append(
                    {
                        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                        "dataset": "msp_podcast",
                        "dataset_release": "R1.10",
                        "utterance_id": utterance,
                        "audio_relpath": relpath,
                        "audio_sha256": audio["audio_sha256"],
                        "speaker_id": f"speaker_{split}_{item}",
                        "speaker_id_status": "known",
                        "group_id": f"speaker_{split}_{item}",
                        "session_id": "podcast",
                        "source_split": source_split,
                        "split": split,
                        "split_version": MSP_SPLIT_VERSION,
                        "original_emotion": "A",
                        "mapped_emotion": "anger",
                        "class_index": 0,
                        "mapping_version": "msp_podcast_r1_10_primary_v1",
                        "included": True,
                        "exclusion_reasons": [],
                        "approximate_mapping": False,
                        **audio,
                    }
                )
        write_manifest(rows, self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def test_resampling_and_multichannel_rejection(self):
        mono = self.root / "mono48.wav"
        sf.write(mono, np.zeros(4800, dtype=np.float32), 48000)
        result = load_audio_16k_mono(mono)
        self.assertEqual(len(result), 1600)
        stereo = self.root / "stereo.wav"
        sf.write(stereo, np.zeros((1600, 2), dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "mono"):
            load_audio_16k_mono(stereo)
        estimate = estimate_full_extraction(
            10.0,
            {"extraction_realtime_factor": 0.5, "feature_bytes_per_audio_second": 1000.0},
        )
        self.assertEqual(estimate["estimated_extraction_seconds"], 5.0)
        self.assertEqual(estimate["required_bytes_with_margin"], 12000)
        self.assertTrue(disk_capacity_gate(self.root, 1)["passes"])

    def test_multiple_shards_mmap_resume_and_partial_recovery(self):
        first = extract_feature_cache(
            self.manifest,
            self.audio_root,
            self.cache,
            FakeEncoder(),
            max_shard_frames=7,
        )
        self.assertEqual(first["extracted"], 6)
        self.assertEqual(first["utterances"], 6)
        self.assertGreaterEqual(sum(item["shards"] for item in first["splits"].values()), 3)
        store = ShardedFeatureStore(self.cache, self.manifest)
        feature = store.get("train_0")
        self.assertEqual(feature.shape, (4, 4))
        self.assertTrue(any(isinstance(array, np.memmap) for array in store._arrays.values()))

        partial = self.cache / "msp_podcast" / "train" / "orphan.partial"
        partial.write_text("incomplete", encoding="utf-8")
        second = extract_feature_cache(
            self.manifest,
            self.audio_root,
            self.cache,
            FakeEncoder(),
            max_shard_frames=7,
        )
        self.assertEqual(second["extracted"], 0)
        self.assertEqual(second["skipped"], 6)
        self.assertEqual(second["removed_partials"], 1)

        success = self.cache / "msp_podcast" / "validation" / "_SUCCESS"
        success.unlink()
        resumed = extract_feature_cache(
            self.manifest,
            self.audio_root,
            self.cache,
            FakeEncoder(),
            max_shard_frames=7,
        )
        self.assertEqual(resumed["extracted"], 0)
        self.assertTrue(success.is_file())

    def test_corruption_and_metadata_mismatch_are_rejected(self):
        extract_feature_cache(self.manifest, self.audio_root, self.cache, FakeEncoder(), max_shard_frames=7)
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_cache(self.cache, self.manifest, expected_signature={"feature_dim": 99})
        shard = next(self.cache.glob("msp_podcast/train/shard-*.npy"))
        with shard.open("ab") as destination:
            destination.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_cache(self.cache, self.manifest)

    def test_encoder_dimension_and_audio_hash_are_enforced(self):
        class BadDimension(FakeEncoder):
            def extract(self, waveform):
                return np.zeros((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "dimension"):
            extract_feature_cache(self.manifest, self.audio_root, self.cache, BadDimension(), max_shard_frames=7)
        # A fresh cache reaches the audio contract before extraction.
        altered = self.audio_root / "Audio" / "train_0.wav"
        sf.write(altered, np.ones(1600, dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "audio hash mismatch"):
            extract_feature_cache(
                self.manifest,
                self.audio_root,
                self.root / "hash_cache",
                FakeEncoder(),
                max_shard_frames=7,
            )


if __name__ == "__main__":
    unittest.main()
