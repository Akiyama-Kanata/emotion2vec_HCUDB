"""Verify class-weight math, checkpoint contracts and MSP comparisons without optimization."""

import copy
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np

from ser_pipeline.audio import sha256_file
from ser_pipeline.cache import ShardedFeatureStore
from ser_pipeline.checkpoints import decoder_signature, restore_resume, save_decoder_checkpoint
from ser_pipeline.evaluation import evaluation_set_signature
from ser_pipeline.model import BaseModel
from ser_pipeline.notebook_api import display_training_history, load_saved_summary, make_demo_artifacts, plot_training_losses, plot_training_scores
from ser_pipeline.study import load_msp_comparison_baselines, run_msp_loss_comparison
from ser_pipeline.training import (
    TrainingConfig, evaluate_loader_metrics, make_loader, train_decoder, train_one_epoch, training_loss_config,
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
        class FixedModel(torch.nn.Module):
            def forward(self, *args):
                return logits
        model = FixedModel()
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

    def test_imbalanced_scores_and_both_loss_aggregations_include_small_final_batch(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 3])
        predictions = torch.tensor([0, 1, 0, 1, 2, 2, 0])
        logits = torch.zeros(7, 4)
        logits[torch.arange(7), predictions] = torch.tensor([2., 1., 3., 2., 4., 1., 5.])
        batches = [
            {"net_input": {"feats": logits[start:start + 3, None].clone().requires_grad_(),
                           "padding_mask": torch.zeros(len(labels[start:start + 3]), 1, dtype=torch.bool)},
             "labels": labels[start:start + 3]}
            for start in range(0, 7, 3)
        ]
        class FixedModel(torch.nn.Module):
            def forward(self, features, mask):
                return features[:, 0]
        model = FixedModel()
        metrics = evaluate_loader_metrics(model, batches, torch.device("cpu"))
        self.assertAlmostEqual(metrics["uar"], (2 / 3 + 1 / 2 + 1) / 4)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 1 / 2 + 2 / 3) / 4)
        self.assertEqual(metrics["accuracy"], 4 / 7)
        probabilities = torch.softmax(logits, -1).numpy().astype(np.float64)
        nll = -np.log(np.clip(probabilities[np.arange(7), labels.numpy()], 1e-12, 1))
        self.assertEqual(metrics["loss"], nll.mean())
        self.assertNotAlmostEqual(metrics["loss"], np.mean([nll[:3].mean(), nll[3:6].mean(), nll[6:].mean()]))
        for weights in (None, [7 / 12, 7 / 8, 7 / 4, 7 / 4]):
            observed = train_one_epoch(model, Mock(), batches, torch.device("cpu"), class_weights=weights)
            sample_weights = np.ones(7) if weights is None else np.array(weights)[labels.numpy()]
            expected = np.mean([(nll[start:start + 3] * sample_weights[start:start + 3]).sum() / sample_weights[start:start + 3].sum()
                                for start in range(0, 7, 3)])
            self.assertAlmostEqual(observed, expected, places=6)


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
        config = TrainingConfig(epochs=2, device="cpu", hidden_dim=4, class_weighting="balanced")
        train_metrics = [{**self.metrics, "uar": .7}, {**self.metrics, "uar": .9}]
        validation_metrics = [self.metrics, {**self.metrics, "uar": .4}]
        output = io.StringIO()
        with patch("ser_pipeline.training.train_one_epoch", return_value=1.) as train, patch(
            "ser_pipeline.training.evaluate_loader_metrics",
            side_effect=[train_metrics[0], validation_metrics[0], train_metrics[1], validation_metrics[1]],
        ) as evaluate, redirect_stdout(output):
            # Timing keys are normally supplied by timed_batches in the mocked functions.
            def fake_epoch(*args, **kwargs):
                kwargs["timings"].update(batch_prepare_seconds=0., compute_seconds=0.)
                return 1.
            train.side_effect = fake_epoch
            result = train_decoder(self.artifact.manifest_path, self.artifact.cache_root, "msp_podcast", self.root / "train", config, training_stage="msp_train", store=self.store)
        self.assertEqual(train.call_args.kwargs["class_weights"], [1.] * 4)
        self.assertEqual(result["loss_config"]["class_weighting"], "balanced")
        self.assertEqual([call.args[1].dataset.split for call in evaluate.call_args_list],
                         ["train", "validation", "train", "validation"])
        self.assertIsNot(train.call_args.args[2], evaluate.call_args_list[0].args[1])
        self.assertEqual(result["best_epoch"], 1)  # Selection uses validation, not improving train scores.
        self.assertEqual(result["best_training_metrics"], train_metrics[0])
        self.assertEqual([entry["train"] for entry in result["history"]], train_metrics)
        self.assertEqual([entry["validation"] for entry in result["history"]], validation_metrics)
        self.assertEqual(self.store.validation_count, 1)
        self.assertIn("train          0.7000   0.4000", output.getvalue())
        self.assertIn("validation     0.5000   0.4000", output.getvalue())
        self.assertIn("accuracy（参考） train=0.6000  validation=0.6000", output.getvalue())
        self.assertIn("best epoch=1", output.getvalue())
        self.assertNotIn("train_loss=", output.getvalue())
        self.assertIn("train_eval=", output.getvalue())
        from ser_pipeline.checkpoints import load_decoder_checkpoint
        self.assertEqual(load_decoder_checkpoint(result["best_checkpoint"])["loss_config"], result["loss_config"])
        self.assertEqual(load_decoder_checkpoint(result["resume_checkpoint"])["history"], result["history"])
        for key in ("best_checkpoint", "resume_checkpoint"):
            payload = load_decoder_checkpoint(result[key])
            self.assertEqual(payload["history_metadata"], result["history_metadata"])
            self.assertEqual(payload["best_training_metrics"], result["best_training_metrics"])
        self.assertNotIn("history_metadata", result["loss_config"])

    def test_best_selection_uses_full_precision_and_keeps_earliest_exact_tie(self):
        validations = [
            {**self.metrics, "uar": .50001, "macro_f1": .9, "loss": .1},
            {**self.metrics, "uar": .50002, "macro_f1": .1, "loss": 9.},
            {**self.metrics, "uar": .50002, "macro_f1": .2, "loss": 10.},
            {**self.metrics, "uar": .50002, "macro_f1": .2, "loss": 8.},
            {**self.metrics, "uar": .50002, "macro_f1": .2, "loss": 8.},
        ]
        def fake_epoch(*args, **kwargs):
            kwargs["timings"].update(batch_prepare_seconds=0., compute_seconds=0.)
            return 1.
        def evaluate(model, loader, device, **kwargs):
            # Train score deliberately has a different best epoch.
            return self.metrics if loader.dataset.split == "train" else validations.pop(0)
        with patch("ser_pipeline.training.train_one_epoch", side_effect=fake_epoch), patch(
            "ser_pipeline.training.evaluate_loader_metrics", side_effect=evaluate
        ), redirect_stdout(io.StringIO()):
            result = train_decoder(self.artifact.manifest_path, self.artifact.cache_root, "msp_podcast", self.root / "ties",
                                   TrainingConfig(epochs=5, device="cpu", hidden_dim=4), training_stage="msp_train", store=self.store)
        self.assertEqual(result["best_epoch"], 4)
        self.assertEqual(result["best_validation_metrics"]["uar"], .50002)
        from ser_pipeline.checkpoints import load_decoder_checkpoint
        self.assertEqual(len(load_decoder_checkpoint(result["best_checkpoint"])["history"]), 4)
        self.assertEqual(load_decoder_checkpoint(result["resume_checkpoint"])["history"], result["history"])

    def test_train_scoring_preserves_weights_rng_and_training_order(self):
        training_loader = make_loader(self.store, "msp_podcast", "train", batch_size=8, shuffle=True, seed=42)
        scoring_loader = make_loader(self.store, "msp_podcast", "train", batch_size=8, shuffle=False, seed=42)
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        random_state = torch.get_rng_state().clone()
        loader_state = training_loader.generator.get_state().clone()
        observed_modes = []
        hook = self.model.register_forward_pre_hook(
            lambda model, args: observed_modes.append((model.training, torch.is_grad_enabled()))
        )
        try:
            metrics = evaluate_loader_metrics(self.model, scoring_loader, torch.device("cpu"))
        finally:
            hook.remove()
        self.assertTrue(observed_modes)
        self.assertTrue(all(mode == (False, False) for mode in observed_modes))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        torch.testing.assert_close(self.model.state_dict(), weights, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), random_state))
        self.assertTrue(torch.equal(training_loader.generator.get_state(), loader_state))
        self.assertEqual(scoring_loader.dataset.ids, self.store.utterance_ids(dataset="msp_podcast", split="train"))
        for score in ("uar", "macro_f1", "wa"):
            self.assertGreaterEqual(metrics[score], 0.)
            self.assertLessEqual(metrics[score], 1.)

    def test_epoch_scores_share_the_model_after_final_update_and_forward_only_in_eval(self):
        snapshots, calls, modes = [], [], []
        current_epoch = 0
        def no_optimization(model, optimizer, loader, device, **kwargs):
            nonlocal current_epoch
            current_epoch += 1
            kwargs["timings"].update(batch_prepare_seconds=0., compute_seconds=0.)
            snapshots.append(copy.deepcopy(model.state_dict()))
            return .123456789
        def score(model, loader, device, **kwargs):
            calls.append((current_epoch, loader.dataset.split, id(model)))
            torch.testing.assert_close(model.state_dict(), snapshots[-1], rtol=0, atol=0)
            hook = model.register_forward_pre_hook(lambda module, args: modes.append((module.training, torch.is_grad_enabled())))
            try:
                result = evaluate_loader_metrics(model, loader, device, **kwargs)
            finally:
                hook.remove()
            torch.testing.assert_close(model.state_dict(), snapshots[-1], rtol=0, atol=0)
            return result
        with patch("ser_pipeline.training.train_one_epoch", side_effect=no_optimization), patch(
            "ser_pipeline.training.evaluate_loader_metrics", side_effect=score
        ), redirect_stdout(io.StringIO()):
            training = train_decoder(self.artifact.manifest_path, self.artifact.cache_root, "msp_podcast", self.root / "fixed-epochs",
                                     TrainingConfig(device="cpu", epochs=2, batch_size=3, hidden_dim=4, dropout=.4), training_stage="msp_train", store=self.store)
        self.assertEqual([(epoch, split) for epoch, split, _ in calls], [(1, "train"), (1, "validation"), (2, "train"), (2, "validation")])
        self.assertEqual(len({identity for _, _, identity in calls}), 1)
        self.assertEqual(len(modes), 10)  # Three train and two validation batches per epoch.
        self.assertTrue(all(mode == (False, False) for mode in modes))
        self.assertEqual([row["train_loss"] for row in training["history"]], [.123456789, .123456789])

    def test_scoring_restores_all_modes_rng_gradients_buffers_and_optimizer_on_failure(self):
        class StochasticProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.BatchNorm1d(8)
                self.dropout = torch.nn.Dropout(.6)
                self.linear = torch.nn.Linear(8, 4)
                self.fail = False
            def forward(self, feats, mask):
                random.random()
                np.random.random()
                torch.rand(2)
                torch.rand(2, device=feats.device)
                result = self.linear(self.dropout(self.norm(feats[:, 0])))
                if self.fail:
                    raise RuntimeError("synthetic forward failure")
                return result

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for failure in (False, True):
                with self.subTest(device=device, failure=failure):
                    model = StochasticProbe().to(device)
                    model.norm.eval()  # Preserve a mixed hierarchy, not just the root flag.
                    model.fail = failure
                    optimizer = torch.optim.AdamW(model.parameters())
                    for parameter in model.parameters():
                        parameter.grad = torch.full_like(parameter, .25)
                        optimizer.state[parameter] = {"step": torch.tensor(3.), "exp_avg": torch.ones_like(parameter), "exp_avg_sq": torch.ones_like(parameter)}
                    states = copy.deepcopy(model.state_dict())
                    gradients = [p.grad.clone() for p in model.parameters()]
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    modes = [module.training for module in model.modules()]
                    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
                    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
                    train = make_loader(self.store, "msp_podcast", "train", batch_size=3, shuffle=True, seed=42)
                    expected = make_loader(self.store, "msp_podcast", "train", batch_size=3, shuffle=True, seed=42)
                    scoring = make_loader(self.store, "msp_podcast", "train", batch_size=3, shuffle=False, seed=42)
                    self.assertEqual(scoring.num_workers, 0)
                    self.assertFalse(scoring.drop_last)
                    self.assertIsInstance(scoring.sampler, torch.utils.data.SequentialSampler)
                    generator_state = train.generator.get_state().clone()
                    observed = []
                    handles = [module.register_forward_pre_hook(lambda module, args: observed.append((module.training, torch.is_grad_enabled()))) for module in model.modules()]
                    try:
                        if failure:
                            with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                                evaluate_loader_metrics(model, scoring, torch.device(device))
                        else:
                            evaluate_loader_metrics(model, scoring, torch.device(device))
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertTrue(observed)
                    self.assertTrue(all(state == (False, False) for state in observed))
                    self.assertEqual(modes, [module.training for module in model.modules()])
                    self.assertEqual(random.getstate(), rng[0])
                    np.testing.assert_equal(np.random.get_state(), rng[1])
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))
                    if cuda_rng is not None:
                        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_rng))
                    torch.testing.assert_close(model.state_dict(), states, rtol=0, atol=0)
                    torch.testing.assert_close([p.grad for p in model.parameters()], gradients, rtol=0, atol=0)
                    torch.testing.assert_close(optimizer.state_dict(), optimizer_state, rtol=0, atol=0)
                    self.assertTrue(torch.equal(train.generator.get_state(), generator_state))
                    self.assertEqual([batch["utterance_ids"] for batch in train], [batch["utterance_ids"] for batch in expected])

    def test_score_plot_uses_recorded_scores_and_leaves_legacy_train_missing(self):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        training = copy.deepcopy(self.parent)
        training["history"][1]["train"] = {"wa": .8, "uar": .7, "macro_f1": .65}
        output = self.root / "scores.png"
        with patch("matplotlib.pyplot.show"), redirect_stdout(io.StringIO()):
            figure = plot_training_scores(training, output_path=output)
        self.assertTrue(output.is_file())
        self.assertEqual(len(figure.axes), 3)
        for axis, key in zip(figure.axes, ("uar", "macro_f1", "wa")):
            lines = {line.get_label(): line for line in axis.lines}
            self.assertEqual(list(lines["train"].get_xdata()), [1, 2])
            self.assertTrue(np.isnan(lines["train"].get_ydata()[0]))
            self.assertEqual(lines["train"].get_ydata()[1], training["history"][1]["train"][key])
            self.assertEqual(list(lines["validation"].get_ydata()), [self.metrics[key]] * 2)
            self.assertEqual(list(lines["Best validation epoch"].get_xdata()), [1, 1])

    def test_loss_html_uses_only_saved_history_handles_nulls_and_does_not_write_on_replay(self):
        import matplotlib
        matplotlib.use("Agg")
        from html.parser import HTMLParser

        class Images(HTMLParser):
            def __init__(self):
                super().__init__()
                self.depth = 0
                self.image_depths = []
            def handle_starttag(self, tag, attrs):
                if tag == "details":
                    self.depth += 1
                    self.assert_closed = "open" not in dict(attrs)
                elif tag == "img":
                    self.image_depths.append(self.depth)
            def handle_endtag(self, tag):
                if tag == "details":
                    self.depth -= 1

        training = copy.deepcopy(self.parent)
        training["history"] += [{"epoch": 3, "train": None, "validation": {"loss": None}},
                                 {"epoch": 4, "train": {"loss": .12345678, "uar": None}, "validation": None, "train_loss": None}]
        training["history"][1]["train"] = {"loss": .7654321, "macro_f1": .7}
        summary = self.root / "history.json"
        summary.write_text(json.dumps(training), encoding="utf-8")
        original = summary.read_bytes()
        with patch("ser_pipeline.training.train_decoder", side_effect=AssertionError("no training")), patch(
            "ser_pipeline.training.evaluate_checkpoint", side_effect=AssertionError("no evaluation")
        ), patch("ser_pipeline.cache.ShardedFeatureStore", side_effect=AssertionError("no cache access")), patch("matplotlib.pyplot.show") as show:
            saved = load_saved_summary(summary)
            rendered = display_training_history(saved, display_output=False).data
            figure = plot_training_losses(saved, show=False)
            show.assert_not_called()
        parser = Images()
        parser.feed(rendered)
        self.assertEqual(parser.image_depths, [0, 1])
        self.assertTrue(parser.assert_closed)
        self.assertIn("未記録", rendered)
        self.assertIn("0.1235", rendered)
        self.assertNotIn("<script", rendered)
        self.assertEqual(summary.read_bytes(), original)
        self.assertEqual(saved, training)
        comparison_lines = {line.get_label(): line for line in figure.axes[0].lines}
        np.testing.assert_equal(comparison_lines["train"].get_ydata(), [np.nan, .7654321, np.nan, .12345678])
        np.testing.assert_equal(comparison_lines["validation"].get_ydata(), [1., 1., np.nan, np.nan])
        optimization_line = next(line for line in figure.axes[1].lines if line.get_label() == "train_loss")
        np.testing.assert_equal(optimization_line.get_ydata(), [1., 1., np.nan, np.nan])
        self.assertFalse(list(self.root.glob("*.png")))
        gapped = {**saved, "history": [saved["history"][0], saved["history"][3]]}
        gaps_figure = plot_training_losses(gapped, show=False)
        gapped_line = next(line for line in gaps_figure.axes[0].lines if line.get_label() == "train")
        self.assertEqual(list(gapped_line.get_xdata()), [1, 2, 3, 4])
        np.testing.assert_equal(gapped_line.get_ydata(), [np.nan, np.nan, np.nan, .12345678])
        display_training_history(saved, save_plots=True, display_output=False)
        self.assertTrue(self.checkpoint.with_suffix('.scores.png').is_file())
        self.assertTrue(self.checkpoint.with_suffix('.losses.png').is_file())
        for history in (None, []):
            self.assertIn("未記録", display_training_history({"history": history}, display_output=False).data)

    def test_new_summary_training_set_provenance_is_checked_without_test_results(self):
        provenance = self.summary["runs"][0]["provenance"]
        provenance.pop("evaluation_sets")
        provenance["training_sets"] = {"msp_podcast": {
            split: evaluation_set_signature(self.artifact.manifest_path, "msp_podcast", split)
            for split in ("train", "validation")
        }}
        self.write_summary()
        self.assertEqual(set(load_msp_comparison_baselines([self.summary_path], self.store, self.config, (42,))), {42})
        provenance["training_sets"]["msp_podcast"]["train"]["utterance_id_sha256"] = "changed"
        self.write_summary()
        with self.assertRaisesRegex(ValueError, "utterance_id_sha256"):
            load_msp_comparison_baselines([self.summary_path], self.store, self.config, (42,))

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
        saved = json.loads(Path(result["summary_path"]).read_text())
        self.assertEqual(saved["runs"][0]["weighted"]["history"], weighted["history"])
        self.assertEqual(saved["training_sets"]["msp_podcast"]["train"]["split"], "train")
        self.assertEqual(saved["runs"][0]["provenance"]["weighted_checkpoint"]["sha256"], sha256_file(weighted["best_checkpoint"]))
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
