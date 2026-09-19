"""SER 特徴キャッシュの作成・再開・検証と本実行前の容量見積りを検証する。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ser_pipeline.audio import inspect_audio, load_audio_16k_mono
from ser_pipeline.cache import (
    ShardedFeatureStore,
    cleanup_uncommitted_cache_fragments,
    inspect_cache_resume,
    validate_cache,
)
from ser_pipeline.contracts import MANIFEST_SCHEMA_VERSION, map_emotion
from ser_pipeline.features import EncoderInfo, extract_feature_cache
from ser_pipeline.manifest import write_manifest
from ser_pipeline.preflight import (
    FeatureExtractionPreflightError,
    disk_capacity_gate,
    estimate_full_extraction,
    preflight_feature_extraction,
    smoke_test_feature_extraction,
)
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

    def test_resume_inspector_finds_only_uncommitted_fragments(self):
        extract_feature_cache(self.manifest, self.audio_root, self.cache, FakeEncoder(), max_shard_frames=7)
        split = self.cache / "msp_podcast" / "train"
        partial = split / "write.partial"
        orphan_npy = split / "shard-99998.npy"
        orphan_index = split / "shard-99998.index.jsonl"
        partial.write_text("partial", encoding="utf-8")
        orphan_npy.write_bytes(b"orphan")
        orphan_index.write_text("orphan\n", encoding="utf-8")

        report = inspect_cache_resume(
            self.cache, self.manifest, expected_dim=4, max_shard_frames=7,
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["committed_utterances"], 6)
        self.assertEqual(report["pending_utterances"], 0)
        self.assertEqual(
            set(report["recoverable_fragments"]),
            {str(path.resolve()) for path in (partial, orphan_npy, orphan_index)},
        )
        removed = cleanup_uncommitted_cache_fragments(
            self.cache, report["recoverable_fragments"],
        )
        self.assertEqual(set(removed), set(report["recoverable_fragments"]))
        self.assertTrue(next(split.glob("shard-*.meta.json")).is_file())

    def test_resume_skips_committed_prefix_after_interruption(self):
        class InterruptingEncoder(FakeEncoder):
            def __init__(self, fail_after=None):
                self.fail_after = fail_after
                self.calls = 0

            def extract(self, waveform):
                self.calls += 1
                if self.fail_after is not None and self.calls > self.fail_after:
                    raise RuntimeError("injected interruption")
                return super().extract(waveform)

        interrupted = InterruptingEncoder(fail_after=2)
        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
            extract_feature_cache(
                self.manifest, self.audio_root, self.cache, interrupted, max_shard_frames=7,
            )
        audit = inspect_cache_resume(
            self.cache, self.manifest, expected_dim=4, max_shard_frames=7,
        )
        self.assertGreaterEqual(audit["committed_utterances"], 1)
        resumed = InterruptingEncoder()
        result = extract_feature_cache(
            self.manifest, self.audio_root, self.cache, resumed, max_shard_frames=7,
        )
        self.assertEqual(resumed.calls, audit["pending_utterances"])
        self.assertEqual(result["skipped"], audit["committed_utterances"])

    def test_preflight_is_read_only_and_smoke_round_trips_without_artifacts(self):
        snapshot = {"checkpoint_sha256": "a" * 64}
        parity = {
            "passed": True,
            "rtol": 1e-5,
            "atol": 1e-6,
            "snapshot": snapshot,
            "extraction": {
                "snapshot": snapshot,
                "extraction_code_version": "test_official_features_v1",
            },
            "extraction_realtime_factor": 0.5,
            "feature_bytes_per_audio_second": 1000.0,
        }
        report = preflight_feature_extraction(
            self.manifest,
            self.audio_root,
            self.cache,
            parity,
            dataset="msp_podcast",
            expected_dim=4,
            max_shard_frames=7,
            expected_label_profile=None,
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["datasets"]["msp_podcast"]["verified_audio"], 6)
        self.assertEqual(report["datasets"]["msp_podcast"]["resume"]["pending_utterances"], 6)
        self.assertFalse(self.cache.exists())

        smoke = smoke_test_feature_extraction(
            self.manifest,
            self.audio_root,
            self.cache,
            FakeEncoder(),
            dataset="msp_podcast",
            sample_size=10,
            expected_dim=4,
            max_shard_frames=7,
        )
        self.assertEqual(smoke["sample_count"], 6)
        self.assertEqual(smoke["sample_utterance_ids"], [
            "train_0", "train_1", "validation_0", "validation_1", "test_0", "test_1",
        ])
        self.assertTrue(smoke["temporary_cache_removed"])
        self.assertFalse(self.cache.exists())
        self.assertEqual(list(self.root.glob(".cache-smoke-*")), [])

    def test_resume_inspector_never_removes_a_committed_corrupt_shard(self):
        extract_feature_cache(self.manifest, self.audio_root, self.cache, FakeEncoder(), max_shard_frames=7)
        shard = next(self.cache.glob("msp_podcast/train/shard-*.npy"))
        with shard.open("ab") as destination:
            destination.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            inspect_cache_resume(self.cache, self.manifest, expected_dim=4, max_shard_frames=7)
        self.assertTrue(shard.is_file())

    def test_all_dataset_preflight_aggregates_late_missing_audio_without_cache_writes(self):
        outer = self.root / "outer_hcudb"
        inner = outer / "HCUDB1"
        hcudb_manifest = self.root / "hcudb.jsonl"
        hcudb_rows = []
        for index, (speaker, split) in enumerate((
            ("FA", "train"), ("FF", "validation"), ("FG", "test"),
        )):
            utterance = f"{speaker}-01-01-1"
            relpath = f"wav/{speaker}/{utterance}.wav"
            path = inner / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.full(1600, 0.02 + index / 100, dtype=np.float32), 16000)
            audio = inspect_audio(path)
            decision = map_emotion("hcudb1", "怒り", label_profile="official6")
            hcudb_rows.append({
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset": "hcudb1",
                "dataset_release": "HCUDB1",
                "utterance_id": utterance,
                "audio_relpath": relpath,
                "speaker_id": speaker,
                "speaker_id_status": "known",
                "group_id": speaker,
                "session_id": "",
                "source_split": "all",
                "split": split,
                "split_version": "hcudb1_speaker_split_v1",
                "original_emotion": "怒り",
                "mapped_emotion": decision.mapped_emotion,
                "class_index": decision.class_index,
                "mapping_version": decision.mapping_version,
                "included": True,
                "exclusion_reasons": [],
                "approximate_mapping": decision.approximate_mapping,
                "source_metadata": {},
                **audio,
            })
        write_manifest(hcudb_rows, hcudb_manifest)
        (inner / hcudb_rows[-1]["audio_relpath"]).unlink()

        snapshot = {"checkpoint_sha256": "a" * 64}
        parity = {
            "passed": True, "rtol": 1e-5, "atol": 1e-6,
            "snapshot": snapshot,
            "extraction": {"snapshot": snapshot, "extraction_code_version": "test_v1"},
            "extraction_realtime_factor": 0.5,
            "feature_bytes_per_audio_second": 1000.0,
        }
        caches = {
            "msp_podcast": self.root / "msp_cache",
            "hcudb1": self.root / "hcudb_cache",
        }
        with self.assertRaises(FeatureExtractionPreflightError) as raised:
            preflight_feature_extraction(
                {"msp_podcast": self.manifest, "hcudb1": hcudb_manifest},
                {"msp_podcast": self.audio_root, "hcudb1": outer},
                caches,
                parity,
                expected_dim=4,
                max_shard_frames=7,
                expected_label_profile=None,
            )
        report = raised.exception.report
        self.assertEqual(report["datasets"]["msp_podcast"]["status"], "ok")
        self.assertEqual(report["datasets"]["hcudb1"]["status"], "error")
        self.assertIn("included audio is missing", report["datasets"]["hcudb1"]["error"])
        self.assertEqual(
            Path(report["datasets"]["hcudb1"]["resolved_audio_root"]), inner.resolve(),
        )
        self.assertTrue(all(not path.exists() for path in caches.values()))


if __name__ == "__main__":
    unittest.main()
