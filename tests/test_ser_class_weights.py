"""Verify class-weight math, checkpoint contracts and MSP comparisons without optimization."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ser_pipeline.audio import sha256_file
from ser_pipeline.cache import ShardedFeatureStore
from ser_pipeline.checkpoints import decoder_signature, restore_resume, save_decoder_checkpoint
from ser_pipeline.evaluation import evaluation_set_signature
from ser_pipeline.model import BaseModel
from ser_pipeline.notebook_api import make_demo_artifacts
from ser_pipeline.study import load_msp_comparison_baselines, run_msp_loss_comparison
from ser_pipeline.training import (
    TrainingConfig, evaluate_loader_metrics, train_decoder, train_one_epoch, training_loss_config,
)


class SerClassWeightsTest(unittest.TestCase):
    def test_weights_use_included_train_utterances_only(self):
        rows = {}
        for split, counts in (("train", [2, 6, 1, 3]), ("validation", [30, 1, 8, 6]), ("test", [1, 80, 4, 5])):
            for label, count in enumerate(counts):
                for i in range(count):
                    rows[f"{split}-{label}-{i}"] = dict(dataset="msp_podcast", split=split, included=True, class_index=label)
        rows["excluded"] = dict(dataset="msp_podcast", split="train", included=False, class_index=0)
        rows["other"] = dict(dataset="hcudb1", split="train", included=True, class_index=0)
        store = SimpleNamespace(records=rows, utterance_ids=lambda **filters: [
            key for key, row in rows.items() if all(row[name] == value for name, value in filters.items())
        ])
        loss = training_loss_config(store, "msp_podcast", "balanced")
        self.assertEqual(loss["train_class_counts"], [2, 6, 1, 3])
        self.assertEqual(loss["class_weights"], [1.5, 0.5, 3.0, 1.0])
        self.assertEqual(loss["label_order"], ["anger", "happy", "sadness", "disgust"])
        self.assertIsNone(training_loss_config(store, "msp_podcast", "none")["class_weights"])
        with self.assertRaisesRegex(ValueError, "none or balanced"):
            training_loss_config(store, "msp_podcast", "typo")
        rows.pop("train-2-0")
        with self.assertRaisesRegex(ValueError, "every class"):
            training_loss_config(store, "msp_podcast", "balanced")

    def test_weighted_loss_and_gradient_match_formula_without_optimizer_updates(self):
        logits = torch.tensor([[2., 0., -1., 1.], [1., 2., 0., -1.]], requires_grad=True)
        labels = torch.tensor([0, 3])
        loader = [{"net_input": {"feats": torch.zeros(2, 1, 4), "padding_mask": torch.zeros(2, 1, dtype=torch.bool)}, "labels": labels}]
        model = Mock(return_value=logits)
        optimizer = Mock()  # No optimizer or model parameters are updated.
        weights = [1., 2., 3., 4.]
        nll = -torch.log_softmax(logits.detach(), dim=-1)[torch.arange(2), labels]
        expected_loss = (nll * torch.tensor([1., 4.])).sum() / 5
        loss = train_one_epoch(model, optimizer, loader, torch.device("cpu"), class_weights=weights)
        self.assertAlmostEqual(loss, expected_loss.item())
        expected_gradient = torch.softmax(logits.detach(), dim=-1)
        expected_gradient[torch.arange(2), labels] -= 1
        expected_gradient *= torch.tensor([[1.], [4.]]) / 5
        torch.testing.assert_close(logits.grad, expected_gradient)
        evaluation = evaluate_loader_metrics(model, loader, torch.device("cpu"))
        self.assertAlmostEqual(evaluation["loss"], nll.mean().item(), places=6)
        logits.grad = None
        plain_loss = train_one_epoch(model, optimizer, loader, torch.device("cpu"))
        self.assertAlmostEqual(plain_loss, nll.mean().item())


class SerLossComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact = make_demo_artifacts(self.root / "data", datasets=("msp_podcast",))["msp_podcast"]
        self.store = ShardedFeatureStore(self.artifact.cache_root, self.artifact.manifest_path)
        self.config = TrainingConfig(epochs=2, device="cpu", hidden_dim=4)
        self.model = BaseModel(input_dim=8, hidden_dim=4)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.metrics = {"uar": .5, "macro_f1": .4, "wa": .6, "loss": 1., "class_metrics": [
            {"class_label": label, "recall": .5} for label in ("anger", "happy", "sadness", "disgust")
        ]}
        self.signature = decoder_signature(self.model, 42, self.store.meta)
        self.checkpoint = self.root / "baseline.pt"
        self.payload = self.save_checkpoint(self.checkpoint)
        self.parent = {
            "dataset": "msp_podcast", "training_stage": "msp_train", "seed": 42,
            "config": asdict(self.config), "best_epoch": 1, "best_checkpoint": str(self.checkpoint),
            "best_validation_metrics": self.metrics,
            "history": [{"epoch": epoch, "train_loss": 1., "validation": self.metrics} for epoch in (1, 2)],
        }
        self.parent["config"].pop("class_weighting")  # Existing study files predate the option.
        self.summary = {"runs": [{"seed": 42, "parent": self.parent, "provenance": {
            "evaluation_sets": {"msp_podcast": evaluation_set_signature(self.artifact.manifest_path, "msp_podcast")},
            "parent_checkpoint": {
                "cache_id": self.store.meta["cache_id"], "sha256": sha256_file(self.checkpoint),
                "id": self.payload["checkpoint_id"],
            },
        }}]}
        self.summary_path = self.root / "study_summary.json"
        self.write_summary()

    def tearDown(self):
        del self.store
        self.temporary.cleanup()

    def save_checkpoint(self, path, loss_config=None):
        return save_decoder_checkpoint(
            path, model=self.model, optimizer=self.optimizer, epoch=1,
            history=[{"epoch": 1}], training_stage="msp_train", signature=self.signature,
            run_id="initialized-only", validation_metrics=self.metrics,
            cache_id=self.store.meta["cache_id"], mapping_versions=self.store.meta["mapping_versions"],
            split_versions=self.store.meta["split_versions"], loss_config=loss_config,
        )

    def write_summary(self):
        self.summary_path.write_text(json.dumps(self.summary), encoding="utf-8")

    def test_resume_rejects_changed_loss_and_supports_legacy_unweighted(self):
        none = training_loss_config(self.store, "msp_podcast", "none")
        balanced = training_loss_config(self.store, "msp_podcast", "balanced")
        restore_resume(self.model, self.optimizer, self.checkpoint, self.signature, "msp_train", expected_loss_config=none)
        with self.assertRaisesRegex(ValueError, "loss configuration mismatch"):
            restore_resume(self.model, self.optimizer, self.checkpoint, self.signature, "msp_train", expected_loss_config=balanced)
        weighted_path = self.root / "weighted.pt"
        self.save_checkpoint(weighted_path, balanced)
        restored = restore_resume(self.model, self.optimizer, weighted_path, self.signature, "msp_train", expected_loss_config=balanced)
        self.assertEqual(restored["loss_config"], balanced)
        with self.assertRaisesRegex(ValueError, "loss configuration mismatch"):
            restore_resume(self.model, self.optimizer, weighted_path, self.signature, "msp_train", expected_loss_config=none)

    def test_training_passes_weights_and_persists_loss_without_optimization(self):
        config = TrainingConfig(epochs=1, device="cpu", hidden_dim=4, class_weighting="balanced")
        with patch("ser_pipeline.training.train_one_epoch", return_value=1.) as train, patch(
            "ser_pipeline.training.evaluate_loader_metrics", return_value=self.metrics,
        ), redirect_stdout(io.StringIO()):
            # Timing keys are normally supplied by timed_batches in the mocked functions.
            def fake_epoch(*args, **kwargs):
                kwargs["timings"].update(batch_prepare_seconds=0., compute_seconds=0.)
                return 1.
            train.side_effect = fake_epoch
            result = train_decoder(self.artifact.manifest_path, self.artifact.cache_root, "msp_podcast", self.root / "train", config, training_stage="msp_train", store=self.store)
        self.assertEqual(train.call_args.kwargs["class_weights"], [1.] * 4)
        self.assertEqual(result["loss_config"]["class_weighting"], "balanced")
        from ser_pipeline.checkpoints import load_decoder_checkpoint
        self.assertEqual(load_decoder_checkpoint(result["best_checkpoint"])["loss_config"], result["loss_config"])

    def test_comparison_reuses_baseline_and_runs_only_weighted_msp(self):
        weighted = copy.deepcopy(self.parent)
        weighted["best_validation_metrics"]["uar"] = .6
        with patch("ser_pipeline.study.train_decoder", return_value=weighted) as train, patch(
            "ser_pipeline.study.evaluate_checkpoint", side_effect=AssertionError("test evaluation must not run"),
        ), redirect_stdout(io.StringIO()):
            result = run_msp_loss_comparison(self.artifact, self.root / "comparison", [self.summary_path], base_config=self.config, store=self.store)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(train.call_args.args[2], "msp_podcast")
        self.assertEqual(train.call_args.args[4].class_weighting, "balanced")
        self.assertNotIn("parent_checkpoint", train.call_args.kwargs)
        self.assertNotIn("resume_checkpoint", train.call_args.kwargs)
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["validation_signature"]["split"], "validation")
        self.assertAlmostEqual(result["runs"][0]["validation_deltas"]["uar"], .1)
        self.assertEqual(self.store.validation_count, 1)
        with self.assertRaisesRegex(ValueError, "not empty"):
            run_msp_loss_comparison(self.artifact, self.root / "comparison", [self.summary_path], base_config=self.config, store=self.store)

    def test_baseline_mismatches_are_rejected_before_training(self):
        original = copy.deepcopy(self.summary)
        for key in ("config", "manifest", "cache", "checkpoint", "epochs"):
            with self.subTest(key=key):
                self.summary = copy.deepcopy(original)
                run = self.summary["runs"][0]
                if key == "config":
                    run["parent"]["config"]["learning_rate"] = .2
                elif key == "manifest":
                    run["provenance"]["evaluation_sets"]["msp_podcast"]["manifest_sha256"] = "changed"
                elif key == "cache":
                    run["provenance"]["parent_checkpoint"]["cache_id"] = "changed"
                elif key == "checkpoint":
                    run["provenance"]["parent_checkpoint"]["sha256"] = "changed"
                else:
                    run["parent"]["history"].pop()
                self.write_summary()
                with self.assertRaises(ValueError), patch("ser_pipeline.study.train_decoder") as train:
                    run_msp_loss_comparison(self.artifact, self.root / "invalid", [self.summary_path], base_config=self.config, store=self.store)
                train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
