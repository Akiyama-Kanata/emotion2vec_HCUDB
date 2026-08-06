import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from iemocap_downstream.notebook_pipeline import (
    SESSION_IDS,
    TrainingConfig,
    evaluate_selected_experiment,
    make_demo_bundle,
    make_session_loaders,
    reload_and_evaluate,
    resolve_device,
    run_five_fold,
    run_one_fold,
    run_validation_experiment,
    select_best_experiment,
    session_split_indices,
    validate_feature_bundle,
)


class IemocapNotebookPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = make_demo_bundle(seed=7, input_dim=8, samples_per_class_session=2)

    def tearDown(self):
        self.temporary.cleanup()

    def test_device_resolution(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto").type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_device("cuda")

    def test_aggregate_validation_and_session_split(self):
        summary = validate_feature_bundle(self.bundle, expected_input_dim=8)
        self.assertEqual(summary["sessions_present"], list(SESSION_IDS))
        self.assertNotIn("labels", summary)
        split = session_split_indices(self.bundle, test_session=3, validation_session=4)
        self.assertTrue(np.all(self.bundle.sessions[split["test"]] == 3))
        self.assertFalse(np.any(self.bundle.sessions[split["train"]] == 3))
        self.assertFalse(np.any(self.bundle.sessions[split["validation"]] == 3))

    def test_one_fold_updates_saves_and_reloads_finite_metrics(self):
        config = TrainingConfig(
            seed=3, device="cpu", epochs=3, batch_size=8, learning_rate=0.002,
            weight_decay=0.0005, hidden_dim=12, dropout=0.1, patience=2,
            test_session=5, validation_session=1,
        )
        checkpoint = self.root / "fold.pt"
        result = run_one_fold(self.bundle, config, checkpoint)
        self.assertTrue(result["optimizer_updated"])
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(np.isfinite(result["history"][0]["train_loss"]))
        self.assertTrue(np.isfinite(list(result["test_metrics"].values())).all())
        self.assertEqual(set(result["split_metrics"]), {"train", "validation", "test"})
        for split_metrics in result["split_metrics"].values():
            self.assertEqual(set(split_metrics), {"loss", "wa", "ua", "macro_f1"})
            self.assertTrue(np.isfinite(list(split_metrics.values())).all())
        self.assertEqual(result["model_selection"], "best_validation_ua")
        self.assertGreaterEqual(result["best_epoch"], 1)
        loaders, _ = make_session_loaders(self.bundle, config)
        _, metadata, metrics = reload_and_evaluate(checkpoint, loaders["test"], "cpu")
        self.assertEqual(metadata["encoder_id"], self.bundle.encoder_id)
        self.assertEqual(metadata["input_dim"], 8)
        self.assertEqual(metadata["class_labels"], list(self.bundle.class_labels))
        self.assertEqual(metadata["seed"], 3)
        self.assertEqual(metadata["test_session"], 5)
        self.assertEqual(metadata["validation_session"], 1)
        self.assertEqual(metadata["hidden_dim"], 12)
        self.assertEqual(metadata["dropout"], 0.1)
        self.assertEqual(metadata["selection_metric"], "validation_ua")
        self.assertEqual(metadata["best_epoch"], result["best_epoch"])
        self.assertEqual(metadata["hyperparameters"]["batch_size"], 8)
        self.assertEqual(metrics, result["reload_metrics"])

    def test_validation_experiment_never_evaluates_test_loader(self):
        config = TrainingConfig(seed=4, device="cpu", epochs=1, batch_size=8, test_session=5, validation_session=1)
        checkpoint = self.root / "validation_only.pt"
        from iemocap_downstream import notebook_pipeline

        original = notebook_pipeline.evaluate_classifier
        evaluated_sessions = []

        def record_sessions(model, loader, device, num_classes):
            indices = loader.dataset.indices
            evaluated_sessions.extend(np.unique(self.bundle.sessions[indices]).tolist())
            return original(model, loader, device, num_classes)

        with mock.patch.object(notebook_pipeline, "evaluate_classifier", side_effect=record_sessions):
            result = run_validation_experiment(self.bundle, config, checkpoint)
        self.assertEqual(evaluated_sessions, [1])
        self.assertNotIn("test_metrics", result)
        self.assertNotIn("split_metrics", result)

    def test_select_best_experiment_tie_break_order_and_base_preference(self):
        def candidate(ua, macro_f1, loss, path):
            return {
                "validation_metrics": {"ua": ua, "macro_f1": macro_f1, "loss": loss, "wa": 0.0},
                "checkpoint_path": str(path),
            }

        base = candidate(70.0, 60.0, 1.0, self.root / "base.pt")
        trial = candidate(71.0, 1.0, 9.0, self.root / "trial.pt")
        self.assertEqual(select_best_experiment({"base": base, "trial": trial})["name"], "trial")
        trial = candidate(70.0, 61.0, 9.0, self.root / "trial.pt")
        self.assertEqual(select_best_experiment({"base": base, "trial": trial})["name"], "trial")
        trial = candidate(70.0, 60.0, 0.9, self.root / "trial.pt")
        self.assertEqual(select_best_experiment({"base": base, "trial": trial})["name"], "trial")
        trial = candidate(70.0, 60.0, 1.0, self.root / "trial.pt")
        self.assertEqual(select_best_experiment({"base": base, "trial": trial})["name"], "base")
        self.assertEqual(select_best_experiment({"trial": trial, "base": base})["name"], "base")

    def test_final_evaluation_has_three_splits_and_four_metrics(self):
        config = TrainingConfig(seed=5, device="cpu", epochs=1, batch_size=8, hidden_dim=10)
        validation = run_validation_experiment(self.bundle, config, self.root / "selected.pt")
        selected = select_best_experiment({"base": validation})
        final = evaluate_selected_experiment(self.bundle, selected)
        self.assertEqual(list(final["split_metrics"]), ["train", "validation", "test"])
        for metrics in final["split_metrics"].values():
            self.assertEqual(set(metrics), {"loss", "wa", "ua", "macro_f1"})
        loaders, _ = make_session_loaders(self.bundle, config)
        _, _, reloaded_test = reload_and_evaluate(self.root / "selected.pt", loaders["test"], "cpu")
        self.assertEqual(reloaded_test, final["split_metrics"]["test"])

    def test_five_fold_summary_is_finite(self):
        config = TrainingConfig(seed=2, device="cpu", epochs=1, batch_size=16)
        summary = run_five_fold(self.bundle, config, self.root / "folds")
        self.assertEqual(len(summary["folds"]), 5)
        self.assertTrue(np.isfinite(list(summary["average"].values())).all())
        self.assertTrue((self.root / "folds" / "five_fold_summary.json").is_file())

    def test_five_fold_selects_between_candidates_before_each_test(self):
        from iemocap_downstream import notebook_pipeline

        events = []

        def validation_result(bundle, config, checkpoint_path):
            name = "trial" if "trial" in Path(checkpoint_path).name else "base"
            events.append(("validation", config.test_session, name))
            return {
                "validation_metrics": {
                    "loss": 0.8 if name == "trial" else 1.0,
                    "wa": 60.0,
                    "ua": 70.0 if name == "trial" else 60.0,
                    "macro_f1": 65.0,
                },
                "checkpoint_path": str(checkpoint_path),
                "test_session": config.test_session,
            }

        def final_evaluation(bundle, selected):
            session = selected["result"]["test_session"]
            events.append(("test", session, selected["name"]))
            metrics = {"loss": 0.5, "wa": 75.0, "ua": 75.0, "macro_f1": 75.0}
            return {"split_metrics": {"train": metrics, "validation": metrics, "test": metrics}}

        configs = {
            "base": TrainingConfig(seed=9, device="cpu", epochs=1),
            "trial": TrainingConfig(seed=9, device="cpu", epochs=1, hidden_dim=12),
        }
        with (
            mock.patch.object(notebook_pipeline, "run_validation_experiment", side_effect=validation_result),
            mock.patch.object(notebook_pipeline, "evaluate_selected_experiment", side_effect=final_evaluation),
        ):
            summary = run_five_fold(self.bundle, configs, self.root / "selected_folds")

        for session in SESSION_IDS:
            self.assertEqual(
                [event for event in events if event[1] == session],
                [("validation", session, "base"), ("validation", session, "trial"), ("test", session, "trial")],
            )
        self.assertTrue(all(fold["selected_experiment"] == "trial" for fold in summary["folds"]))

    def test_lecture_notebook_has_one_line_experiments_and_final_table(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "iemocap_base_downstream_training.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        by_id = {cell["id"]: cell for cell in notebook["cells"]}
        self.assertEqual(
            by_id["base-run"]["source"].strip(),
            "base_result = run_validation_experiment(bundle, hp_base)",
        )
        self.assertEqual(
            by_id["trial-run"]["source"].strip(),
            "trial_result = run_validation_experiment(bundle, hp_trial)",
        )
        all_source = "\n".join(cell["source"] for cell in notebook["cells"])
        self.assertNotIn("run_one_fold", all_source)
        self.assertIn("select_best_experiment({'base': base_result, 'trial': trial_result})", all_source)
        final_source = by_id["final-evaluation"]["source"]
        self.assertEqual(final_source.count("evaluate_selected_experiment"), 1)
        for split in ("train", "validation", "test"):
            self.assertIn(f"'{split}'", final_source)
        for metric in ("loss", "wa", "ua", "macro_f1"):
            self.assertIn(f"'{metric}'", final_source)


if __name__ == "__main__":
    unittest.main()
