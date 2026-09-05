"""Verify deterministic lightweight train monitoring and final full-train results."""

import copy
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ser_pipeline.cache import ShardedFeatureStore
from ser_pipeline.checkpoints import load_decoder_checkpoint
from ser_pipeline.notebook_api import display_training_history, make_demo_artifacts
from ser_pipeline.training import (
    TrainingConfig,
    TrainingMonitoringConfig,
    build_train_monitoring,
    evaluate_loader_metrics,
    train_decoder,
)


def metric_support(metrics):
    return sum(int(row["support"]) for row in metrics["class_metrics"])


class TrainMonitorSelectionTest(unittest.TestCase):
    def test_stratified_hash_selection_is_fixed_unique_and_manifest_ordered(self):
        counts = (2753, 9686, 1854, 1231)  # Current 15,524-item MSP train distribution.
        records = {}
        manifest_ids = []
        for serial in range(max(counts)):
            for class_index, count in enumerate(counts):
                if serial < count:
                    utterance_id = f"msp-{class_index}-{serial:05d}"
                    records[utterance_id] = {
                        "dataset": "msp_podcast", "split": "train", "class_index": class_index,
                    }
                    manifest_ids.append(utterance_id)
        store = SimpleNamespace(
            records=records,
            utterance_ids=lambda **filters: [
                utterance_id for utterance_id in manifest_ids
                if all(records[utterance_id][key] == value for key, value in filters.items())
            ],
        )
        config = TrainingMonitoringConfig(max_epoch_samples=2000, sampling_seed=0)
        first_ids, first = build_train_monitoring(store, "msp_podcast", config)
        second_ids, second = build_train_monitoring(store, "msp_podcast", config)

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first, second)
        self.assertEqual(len(first_ids), 2000)
        self.assertEqual(len(set(first_ids)), 2000)
        self.assertTrue(set(first_ids).issubset(manifest_ids))
        positions = {utterance_id: index for index, utterance_id in enumerate(manifest_ids)}
        self.assertEqual([positions[item] for item in first_ids], sorted(positions[item] for item in first_ids))
        self.assertEqual(sum(first["class_counts"]), 2000)
        self.assertEqual(first["class_counts"], [355, 1248, 239, 158])
        for selected, population in zip(first["class_counts"], counts):
            self.assertLess(abs(selected - 2000 * population / sum(counts)), 1)
        self.assertEqual(first["sampling_method"], "stratified_class_quota_stable_sha256_rank_v1")
        self.assertEqual(len(first["utterance_id_sha256"]), 64)


class TrainMonitoringIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = make_demo_artifacts(
            self.root / "data", datasets=("msp_podcast",)
        )["msp_podcast"]
        self.store = ShardedFeatureStore(self.artifact.cache_root, self.artifact.manifest_path)

    def tearDown(self):
        self.store._arrays.clear()
        gc.collect()
        self.temporary.cleanup()

    def _train(self, name, monitoring, *, epochs=2, resume=None, seed=42, class_weighting="none"):
        return train_decoder(
            self.artifact.manifest_path,
            self.artifact.cache_root,
            "msp_podcast",
            self.root / name,
            TrainingConfig(
                seed=seed, device="cpu", epochs=epochs, batch_size=3, hidden_dim=8,
                dropout=.3, class_weighting=class_weighting,
            ),
            training_stage="msp_train",
            monitoring_config=monitoring,
            resume_checkpoint=resume,
            store=self.store,
        )

    def test_partial_monitor_then_one_full_evaluation_and_atomic_checkpoint_annotation(self):
        calls = []
        updates = []
        original_evaluate = evaluate_loader_metrics
        from ser_pipeline.checkpoints import update_decoder_checkpoint_results as original_update

        def evaluate(model, loader, device, **kwargs):
            calls.append((loader.dataset.split, list(loader.dataset.ids)))
            return original_evaluate(model, loader, device, **kwargs)

        def update(path, **kwargs):
            before = load_decoder_checkpoint(path, map_location=None)
            protected = copy.deepcopy({
                key: before[key]
                for key in ("model_state_dict", "optimizer_state_dict", "epoch", "checkpoint_id")
            })
            self.assertIsNone(before["best_training_metrics"])
            result = original_update(path, **kwargs)
            after = load_decoder_checkpoint(path, map_location=None)
            torch.testing.assert_close(after["model_state_dict"], protected["model_state_dict"], rtol=0, atol=0)
            torch.testing.assert_close(after["optimizer_state_dict"], protected["optimizer_state_dict"], rtol=0, atol=0)
            self.assertEqual(after["epoch"], protected["epoch"])
            self.assertEqual(after["checkpoint_id"], protected["checkpoint_id"])
            updates.append(Path(path))
            return result

        monitoring = TrainingMonitoringConfig(max_epoch_samples=4, sampling_seed=0)
        with patch("ser_pipeline.training.evaluate_loader_metrics", side_effect=evaluate), patch(
            "ser_pipeline.training.update_decoder_checkpoint_results", side_effect=update
        ):
            result = self._train("partial", monitoring)

        self.assertEqual([(split, len(ids)) for split, ids in calls], [
            ("train", 4), ("validation", 4),
            ("train", 4), ("validation", 4),
            ("train", 8),
        ])
        self.assertEqual(len(updates), 2)
        self.assertEqual(metric_support(result["history"][result["best_epoch"] - 1]["train_monitor"]), 4)
        self.assertEqual(metric_support(result["best_training_metrics"]), 8)
        self.assertNotIn("train", result["history"][0])
        self.assertEqual(result["monitoring_config"], {"max_epoch_samples": 4, "sampling_seed": 0})
        self.assertFalse(result["timings"]["best_train_evaluation_reused_from_monitor"])
        self.assertGreater(result["timings"]["best_train_evaluation_seconds"], 0)
        for epoch in result["timings"]["epochs"]:
            self.assertIn("train_monitor_evaluation_seconds", epoch)
            self.assertNotIn("train_evaluation_seconds", epoch)
        for checkpoint in (result["best_checkpoint"], result["resume_checkpoint"]):
            payload = load_decoder_checkpoint(checkpoint)
            self.assertEqual(payload["best_training_metrics"], result["best_training_metrics"])
            self.assertEqual(payload["monitoring_config"], result["monitoring_config"])
            self.assertEqual(payload["train_monitoring"], result["train_monitoring"])
        rendered = display_training_history(result, display_output=False).data
        self.assertIn("train（固定4件・参考）", rendered)
        self.assertIn("bestモデルのtrain全件結果（正式）", rendered)

    def test_full_monitor_skips_duplicate_final_evaluation(self):
        calls = []
        original_evaluate = evaluate_loader_metrics

        def evaluate(model, loader, device, **kwargs):
            calls.append((loader.dataset.split, len(loader.dataset)))
            return original_evaluate(model, loader, device, **kwargs)

        with patch("ser_pipeline.training.evaluate_loader_metrics", side_effect=evaluate):
            result = self._train("full", TrainingMonitoringConfig(max_epoch_samples=2000), epochs=1)
        self.assertEqual(calls, [("train", 8), ("validation", 4)])
        self.assertTrue(result["timings"]["best_train_evaluation_reused_from_monitor"])
        self.assertEqual(result["timings"]["best_train_evaluation_seconds"], 0.0)
        self.assertEqual(metric_support(result["best_training_metrics"]), 8)

    def test_monitoring_does_not_change_weights_batch_order_or_best_epoch(self):
        full = self._train("all-train", None)
        partial = self._train("fixed-monitor", TrainingMonitoringConfig(max_epoch_samples=4))
        self.assertEqual(full["best_epoch"], partial["best_epoch"])
        for key in ("best_checkpoint", "resume_checkpoint"):
            full_payload = load_decoder_checkpoint(full[key])
            partial_payload = load_decoder_checkpoint(partial[key])
            torch.testing.assert_close(
                full_payload["model_state_dict"], partial_payload["model_state_dict"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                full_payload["optimizer_state_dict"], partial_payload["optimizer_state_dict"], rtol=0, atol=0
            )

    def test_fixed_monitor_is_shared_across_training_seeds_and_loss_settings(self):
        monitoring = TrainingMonitoringConfig(max_epoch_samples=4, sampling_seed=0)
        seed_42 = self._train("seed-42-none", monitoring, epochs=1, seed=42)
        seed_44 = self._train(
            "seed-44-balanced", monitoring, epochs=1, seed=44, class_weighting="balanced"
        )
        self.assertEqual(
            seed_42["train_monitoring"]["utterance_id_sha256"],
            seed_44["train_monitoring"]["utterance_id_sha256"],
        )
        self.assertEqual(
            seed_42["train_monitoring"]["class_counts"],
            seed_44["train_monitoring"]["class_counts"],
        )

    def test_resume_requires_exact_monitoring_configuration(self):
        first = self._train("first", TrainingMonitoringConfig(max_epoch_samples=4), epochs=1)
        with self.assertRaisesRegex(ValueError, "monitoring configuration mismatch"):
            self._train(
                "mismatch",
                TrainingMonitoringConfig(max_epoch_samples=5),
                epochs=2,
                resume=first["resume_checkpoint"],
            )
        resumed = self._train(
            "resumed",
            TrainingMonitoringConfig(max_epoch_samples=4),
            epochs=2,
            resume=first["resume_checkpoint"],
        )
        self.assertEqual(len(resumed["history"]), 2)
        self.assertTrue(all("train_monitor" in row and "train" not in row for row in resumed["history"]))

    def test_legacy_checkpoint_resumes_only_as_full_train_history_without_rewriting_source(self):
        first = self._train("new-full", None, epochs=1)
        payload = load_decoder_checkpoint(first["resume_checkpoint"], map_location=None)
        payload.pop("monitoring_config")
        payload.pop("train_monitoring")
        payload.pop("best_train_evaluation_seconds")
        payload.pop("best_train_evaluation_reused_from_monitor")
        payload.pop("history_metadata")
        for row in payload["history"]:
            row["train"] = row.pop("train_monitor")
        legacy_path = self.root / "legacy-last.pt"
        torch.save(payload, legacy_path)
        original = legacy_path.read_bytes()

        resumed = self._train("legacy-resumed", None, epochs=2, resume=legacy_path)

        self.assertEqual(legacy_path.read_bytes(), original)
        self.assertTrue(all("train" in row and "train_monitor" not in row for row in resumed["history"]))
        for checkpoint in (resumed["best_checkpoint"], resumed["resume_checkpoint"]):
            self.assertNotIn("monitoring_config", load_decoder_checkpoint(checkpoint))


if __name__ == "__main__":
    unittest.main()
