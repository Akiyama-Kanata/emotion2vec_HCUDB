"""小規模な人工データで SER 学習・転移・評価の全体経路を検証する。"""

import tempfile
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ser_pipeline.audio import sha256_file
from ser_pipeline.checkpoints import load_decoder_checkpoint
from ser_pipeline.notebook_api import make_demo_artifacts, run_demo_transfer_study
from ser_pipeline.study import DatasetArtifacts, EVALUATION_DATASETS, STUDY_SEEDS, FinalEvaluationTarget, require_formal_epochs, run_final_evaluations
from ser_pipeline.training import CachedFeatureDataset, TrainingConfig, evaluate_loader_metrics, train_decoder, train_one_epoch


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

    def test_training_then_explicit_final_parent_child_evaluations(self):
        summary = run_demo_transfer_study(self.root, seeds=(42,), epochs=1)
        self.assertTrue(Path(summary["summary_path"]).is_file())
        self.assertEqual(summary["seeds"], [42])
        self.assertEqual(summary["evaluation_datasets"], ["msp_podcast", "hcudb1"])
        run = summary["runs"][0]
        self.assertFalse(summary["test_evaluated"])
        self.assertNotIn("before", run)
        self.assertNotIn("after", run)
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
        self.assertEqual(set(provenance["training_sets"]), set(EVALUATION_DATASETS))
        artifacts = {dataset: DatasetArtifacts(self.root / "artifacts" / dataset / "manifest.jsonl",
                                              self.root / "artifacts" / dataset / "cache") for dataset in EVALUATION_DATASETS}
        targets = [FinalEvaluationTarget(stage, path, sha256_file(path), dataset)
                   for stage, path in (("parent", parent_path), ("child", child_path)) for dataset in EVALUATION_DATASETS]
        final = run_final_evaluations(artifacts, targets, self.root / "final", device="cpu", batch_size=4)
        for dataset in EVALUATION_DATASETS:
            before, after = [evaluation["result"] for evaluation in final["evaluations"] if evaluation["target"]["dataset"] == dataset]
            self.assertEqual(before["set_signature"], after["set_signature"])
            self.assertEqual(before["cache_id"], after["cache_id"])
            for stage in (before, after):
                metrics = stage["metrics_4class"]
                for name in ("accuracy", "wa", "uar", "macro_f1"):
                    self.assertGreaterEqual(metrics[name], 0.0)
                    self.assertLessEqual(metrics[name], 1.0)
                self.assertEqual(len(stage["predictions"][0]["probabilities"]), 4)
        saved_summary = json.loads(Path(summary["summary_path"]).read_text())
        for stage in ("parent", "child"):
            training = run[stage]
            self.assertEqual(training["history"], saved_summary["runs"][0][stage]["history"])
            self.assertEqual(training["history"], load_decoder_checkpoint(training["resume_checkpoint"])["history"])
            self.assertEqual(training["history_metadata"], saved_summary["runs"][0][stage]["history_metadata"])
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

    def test_train_scoring_does_not_change_multi_epoch_optimization(self):
        """USER-RUN: exact CPU comparison with/without train scoring for all seeds/losses."""
        from ser_pipeline.checkpoints import save_decoder_checkpoint
        from ser_pipeline.cache import ShardedFeatureStore
        from ser_pipeline.manifest import write_manifest
        from ser_pipeline.notebook_api import _demo_rows, _record, _write_demo_cache

        # Use unequal training counts so balanced weights actually differ from one.
        rows = [row for row in _demo_rows("msp_podcast") if row["split"] != "train"]
        rows.extend(_record("msp_podcast", "train", label, serial)
                    for label, count in enumerate((2, 4, 1, 3)) for serial in range(count))
        artifact = DatasetArtifacts(self.root / "scoring-data" / "manifest.jsonl", self.root / "scoring-data" / "cache")
        write_manifest(rows, artifact.manifest_path)
        _write_demo_cache(artifact.cache_root, artifact.manifest_path, rows, feature_dim=8)
        store = ShardedFeatureStore(artifact.cache_root, artifact.manifest_path)
        for seed in (42, 43, 44):
            for weighting in ("none", "balanced"):
                for dropout in (0., .3):
                    with self.subTest(seed=seed, weighting=weighting, dropout=dropout):
                        traces = []
                        for enabled in (False, True):
                            epochs, orders = [], []
                            def score(model, loader, device, **kwargs):
                                if not enabled and loader.dataset.split == "train":
                                    return {"uar": 0., "macro_f1": 0., "wa": 0., "loss": None}
                                return evaluate_loader_metrics(model, loader, device, **kwargs)
                            def epoch(model, optimizer, loader, device, **kwargs):
                                order = []
                                def recorded_batches():
                                    for batch in loader:
                                        order.extend(batch["utterance_ids"])
                                        yield batch
                                loss = train_one_epoch(model, optimizer, recorded_batches(), device, **kwargs)
                                orders.append(order)
                                return loss
                            def save(path, **kwargs):
                                payload = save_decoder_checkpoint(path, **kwargs)
                                if kwargs["selection"] == "last":
                                    epochs.append(copy.deepcopy({key: payload[key] for key in (
                                        "model_state_dict", "optimizer_state_dict", "validation_metrics", "best_epoch",
                                    )}))
                                return payload
                            config = TrainingConfig(seed=seed, device="cpu", epochs=3, batch_size=3, hidden_dim=8,
                                                    dropout=dropout, class_weighting=weighting)
                            with patch("ser_pipeline.training.evaluate_loader_metrics", side_effect=score), patch(
                                "ser_pipeline.training.train_one_epoch", side_effect=epoch
                            ), patch("ser_pipeline.training.save_decoder_checkpoint", side_effect=save):
                                result = train_decoder(artifact.manifest_path, artifact.cache_root, "msp_podcast",
                                                       self.root / f"trace-{seed}-{weighting}-{dropout}-{enabled}", config,
                                                       training_stage="msp_train", store=store)
                            traces.append((epochs, orders, result))
                        without, with_scoring = traces
                        self.assertEqual(len(without[0]), len(with_scoring[0]))
                        for without_epoch, with_epoch in zip(without[0], with_scoring[0]):
                            torch.testing.assert_close(
                                without_epoch["model_state_dict"], with_epoch["model_state_dict"], rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                without_epoch["optimizer_state_dict"], with_epoch["optimizer_state_dict"], rtol=0, atol=0
                            )
                            self.assertEqual(
                                without_epoch["validation_metrics"], with_epoch["validation_metrics"]
                            )
                            self.assertEqual(without_epoch["best_epoch"], with_epoch["best_epoch"])
                        self.assertEqual(without[1], with_scoring[1])
                        self.assertEqual(without[2]["best_epoch"], with_scoring[2]["best_epoch"])
                        for field in ("train_loss", "validation"):
                            self.assertEqual([row[field] for row in without[2]["history"]], [row[field] for row in with_scoring[2]["history"]])
        store._arrays.clear()


if __name__ == "__main__":
    unittest.main()
