import tempfile
import unittest
from pathlib import Path

from ser_pipeline.audio import sha256_file
from ser_pipeline.checkpoints import load_decoder_checkpoint
from ser_pipeline.notebook_api import make_demo_artifacts, run_demo_transfer_study
from ser_pipeline.study import STUDY_SEEDS
from ser_pipeline.training import TrainingConfig, train_decoder


class SerEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_cache_parent_three_evaluations_child_same_evaluations(self):
        self.assertEqual(STUDY_SEEDS, (42, 43, 44))
        summary = run_demo_transfer_study(self.root, seeds=(42,), epochs=1)
        self.assertTrue(Path(summary["summary_path"]).is_file())
        self.assertEqual(summary["seeds"], [42])
        run = summary["runs"][0]
        self.assertEqual(set(run["before"]), {"msp_podcast", "hcudb1", "iemocap"})
        self.assertEqual(set(run["after"]), {"msp_podcast", "hcudb1", "iemocap"})
        parent_path = Path(run["parent"]["best_checkpoint"])
        child_path = Path(run["child"]["best_checkpoint"])
        parent = load_decoder_checkpoint(parent_path)
        child = load_decoder_checkpoint(child_path)
        self.assertEqual(parent["training_stage"], "msp_train")
        self.assertEqual(child["training_stage"], "hcudb_continue")
        self.assertEqual(child["parent_checkpoint_id"], parent["checkpoint_id"])
        self.assertEqual(child["parent_checkpoint_sha256"], sha256_file(parent_path))
        for dataset in run["before"]:
            before = run["before"][dataset]["result"]
            after = run["after"][dataset]["result"]
            self.assertEqual(before["set_signature"], after["set_signature"])
            for stage in (before, after):
                metrics = stage["metrics_4class"]
                for name in ("accuracy", "wa", "uar", "macro_f1"):
                    self.assertGreaterEqual(metrics[name], 0.0)
                    self.assertLessEqual(metrics[name], 1.0)
                self.assertEqual(len(stage["predictions"][0]["probabilities"]), 4)
        self.assertIsNotNone(run["before"]["iemocap"]["result"]["metrics_primary_3class"])

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


if __name__ == "__main__":
    unittest.main()
