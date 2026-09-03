"""小規模な人工データで SER 学習・転移・評価の全体経路を検証する。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ser_pipeline.audio import sha256_file
from ser_pipeline.checkpoints import load_decoder_checkpoint
from ser_pipeline.notebook_api import make_demo_artifacts, run_demo_transfer_study
from ser_pipeline.study import EVALUATION_DATASETS, STUDY_SEEDS, require_formal_epochs
from ser_pipeline.training import CachedFeatureDataset, TrainingConfig, train_decoder


class SerEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_study_contract_and_formal_epoch_gate_do_not_train(self):
        self.assertEqual(EVALUATION_DATASETS, ("msp_podcast", "hcudb1"))
        self.assertEqual(STUDY_SEEDS, (42, 43, 44))
        with self.assertRaisesRegex(ValueError, "set explicitly"):
            require_formal_epochs(None)
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                require_formal_epochs(invalid)
        self.assertEqual(require_formal_epochs(7), 7)

    def test_manifest_cache_parent_two_evaluations_child_same_evaluations(self):
        summary = run_demo_transfer_study(self.root, seeds=(42,), epochs=1)
        self.assertTrue(Path(summary["summary_path"]).is_file())
        self.assertEqual(summary["seeds"], [42])
        self.assertEqual(summary["evaluation_datasets"], ["msp_podcast", "hcudb1"])
        run = summary["runs"][0]
        self.assertEqual(set(run["before"]), {"msp_podcast", "hcudb1"})
        self.assertEqual(set(run["after"]), {"msp_podcast", "hcudb1"})
        parent_path = Path(run["parent"]["best_checkpoint"])
        child_path = Path(run["child"]["best_checkpoint"])
        parent = load_decoder_checkpoint(parent_path)
        child = load_decoder_checkpoint(child_path)
        self.assertEqual(parent["training_stage"], "msp_train")
        self.assertEqual(child["training_stage"], "hcudb_continue")
        self.assertEqual(child["parent_checkpoint_id"], parent["checkpoint_id"])
        self.assertEqual(child["parent_checkpoint_sha256"], sha256_file(parent_path))
        provenance = run["provenance"]
        self.assertEqual(provenance["parent_checkpoint"]["id"], parent["checkpoint_id"])
        self.assertEqual(provenance["parent_checkpoint"]["sha256"], sha256_file(parent_path))
        self.assertEqual(provenance["child_checkpoint"]["id"], child["checkpoint_id"])
        self.assertEqual(provenance["child_checkpoint"]["sha256"], sha256_file(child_path))
        self.assertEqual(set(provenance["evaluation_sets"]), set(EVALUATION_DATASETS))
        for dataset in run["before"]:
            before = run["before"][dataset]["result"]
            after = run["after"][dataset]["result"]
            self.assertEqual(before["set_signature"], after["set_signature"])
            self.assertEqual(before["cache_id"], after["cache_id"])
            for stage in (before, after):
                metrics = stage["metrics_4class"]
                for name in ("accuracy", "wa", "uar", "macro_f1"):
                    self.assertGreaterEqual(metrics[name], 0.0)
                    self.assertLessEqual(metrics[name], 1.0)
                self.assertEqual(len(stage["predictions"][0]["probabilities"]), 4)
    def test_resume_restores_same_run_while_parent_starts_new_run(self):
        artifacts = make_demo_artifacts(self.root / "artifacts")
        msp = artifacts["msp_podcast"]
        first = train_decoder(
            msp.manifest_path,
            msp.cache_root,
            "msp_podcast",
            self.root / "first",
            TrainingConfig(seed=42, device="cpu", epochs=1, batch_size=4, hidden_dim=8),
            training_stage="msp_train",
        )
        first_last = load_decoder_checkpoint(first["resume_checkpoint"])
        resumed = train_decoder(
            msp.manifest_path,
            msp.cache_root,
            "msp_podcast",
            self.root / "resumed",
            TrainingConfig(seed=42, device="cpu", epochs=2, batch_size=4, hidden_dim=8),
            training_stage="msp_train",
            resume_checkpoint=first["resume_checkpoint"],
        )
        resumed_last = load_decoder_checkpoint(resumed["resume_checkpoint"])
        self.assertEqual(first_last["run_id"], resumed_last["run_id"])
        self.assertEqual(resumed_last["epoch"], 2)
        hcudb = artifacts["hcudb1"]
        child = train_decoder(
            hcudb.manifest_path,
            hcudb.cache_root,
            "hcudb1",
            self.root / "child",
            TrainingConfig(seed=42, device="cpu", epochs=1, batch_size=4, hidden_dim=8),
            training_stage="hcudb_continue",
            parent_checkpoint=first["best_checkpoint"],
        )
        child_payload = load_decoder_checkpoint(child["resume_checkpoint"])
        self.assertNotEqual(first_last["run_id"], child_payload["run_id"])
        resumed_child = train_decoder(
            hcudb.manifest_path,
            hcudb.cache_root,
            "hcudb1",
            self.root / "resumed_child",
            TrainingConfig(seed=42, device="cpu", epochs=2, batch_size=4, hidden_dim=8),
            training_stage="hcudb_continue",
            resume_checkpoint=child["resume_checkpoint"],
        )
        resumed_child_payload = load_decoder_checkpoint(resumed_child["resume_checkpoint"])
        self.assertEqual(resumed_child_payload["parent_checkpoint_id"], child_payload["parent_checkpoint_id"])
        self.assertEqual(resumed_child_payload["parent_checkpoint_sha256"], child_payload["parent_checkpoint_sha256"])

    def test_direct_copy_training_matches_legacy_history_and_weights(self):
        """User-run optimization regression; this test invokes train_decoder."""
        artifacts = make_demo_artifacts(self.root / "artifacts", datasets=("msp_podcast",))
        artifact = artifacts["msp_podcast"]

        class LegacyDataset(CachedFeatureDataset):
            def __getitem__(self, index):
                features, label, utterance_id = super().__getitem__(index)
                return torch.from_numpy(features.copy()).float(), label, utterance_id

        config = TrainingConfig(seed=42, device="cpu", epochs=2, batch_size=3, hidden_dim=8)
        optimized = train_decoder(
            artifact.manifest_path, artifact.cache_root, "msp_podcast", self.root / "direct",
            config, training_stage="msp_train",
        )
        with patch("ser_pipeline.training.CachedFeatureDataset", LegacyDataset):
            legacy = train_decoder(
                artifact.manifest_path, artifact.cache_root, "msp_podcast", self.root / "legacy",
                config, training_stage="msp_train",
            )
        self.assertEqual(optimized["history"], legacy["history"])
        self.assertEqual(optimized["best_epoch"], legacy["best_epoch"])
        for key in ("best_checkpoint", "resume_checkpoint"):
            optimized_weights = load_decoder_checkpoint(optimized[key])["model_state_dict"]
            legacy_weights = load_decoder_checkpoint(legacy[key])["model_state_dict"]
            for name in optimized_weights:
                self.assertTrue(torch.equal(optimized_weights[name], legacy_weights[name]), name)
        self.assertTrue(Path(optimized["timings_path"]).is_file())
        self.assertEqual(len(optimized["timings"]["epochs"]), 2)
        for epoch in optimized["timings"]["epochs"]:
            self.assertGreater(epoch["train"]["batch_prepare_seconds"], 0)
            self.assertGreater(epoch["train"]["compute_seconds"], 0)
            self.assertGreater(epoch["validation_seconds"], 0)
            self.assertGreater(epoch["save_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
