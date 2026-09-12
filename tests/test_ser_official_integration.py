"""Opt-in real-snapshot parity and synthetic cached adaptation integration tests."""

import json
import copy
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from ser_pipeline.audio import load_audio_16k_mono
from ser_pipeline.model import OfficialHeadModel
from ser_pipeline.official import OfficialEmotion2vecEncoder, state_sha256
from ser_pipeline.official import OfficialHead, OfficialLabelSpec
from ser_pipeline.audio import sha256_file
from ser_pipeline.notebook_api import make_demo_artifacts
from ser_pipeline.cache import ShardedFeatureStore, validate_official_cache
from ser_pipeline.training import (
    TrainingConfig,
    TrainingMonitoringConfig,
    train_official_decoder,
    evaluate_official,
)
from ser_pipeline.diagnostics import OfficialTrainingDiagnosticsConfig
from ser_pipeline.checkpoints import load_official_checkpoint
from ser_pipeline.study import run_official_study, run_official_final_evaluations


class OfficialSyntheticIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifacts = make_demo_artifacts(
            self.root / "data", feature_dim=1024,
            datasets=("hcudb1", "msp_podcast"), label_profile="official6",
        )
        spec = OfficialLabelSpec(("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"))
        torch.manual_seed(9)
        proj = torch.nn.Linear(1024, 9)
        self.head = OfficialHead(proj, spec, {
            "revision": "synthetic-test-only", "checkpoint_sha256": "1" * 64, "config_sha256": "2" * 64,
            "tokens_sha256": "3" * 64, "head_sha256": state_sha256({"proj.weight": proj.weight, "proj.bias": proj.bias}),
            "label_spec": spec.as_dict(), "normalize": True, "mask": False, "remove_extra_tokens": True,
        })
        import ser_pipeline.official as official
        hashes = official.implementation_hashes()
        provenance = {"snapshot": self.head.provenance, "extraction_code_version": "ser_official_features_v1",
                      "implementation_sha256": official.implementation_sha256(hashes),
                      "implementation_files_sha256": hashes, "dependencies": {"test": "synthetic"}}
        self.report = {"passed": True, "reference_api": "funasr.AutoModel.generate", "snapshot": self.head.provenance,
                       "extraction": provenance, "rtol": 1e-5, "atol": 1e-6}
        for artifact in self.artifacts.values():
            meta_path = artifact.cache_root / "cache_meta.json"
            meta = json.loads(meta_path.read_text())
            meta.update(encoder_name="emotion2vec_plus_large", encoder_checkpoint_sha256="1" * 64,
                        extraction_code_version="ser_official_features_v1", official_provenance=provenance)
            meta_path.write_text(json.dumps(meta))

    def tearDown(self):
        self.tmp.cleanup()

    def train(self, name, epochs, resume=None):
        artifact = self.artifacts["hcudb1"]
        return train_official_decoder(artifact.manifest_path, artifact.cache_root, self.root / name,
            self.head, self.report, config=TrainingConfig(epochs=epochs, weight_decay=0, batch_size=3, device="cpu"),
            resume_checkpoint=resume)

    def test_resume_exact_and_reject_corruption(self):
        with patch("ser_pipeline.training.evaluate_model", side_effect=AssertionError("test evaluation during training")):
            continuous = self.train("continuous", 3)
            first = self.train("first", 1)
            resumed = self.train("resumed", 3, first["resume_checkpoint"])
        left = load_official_checkpoint(continuous["resume_checkpoint"], self.head)
        right = load_official_checkpoint(resumed["resume_checkpoint"], self.head)
        self.assertEqual(left["history"], right["history"])
        for key, value in left["model_state_dict"].items():
            self.assertTrue(torch.equal(value, right["model_state_dict"][key]))
        best = load_official_checkpoint(resumed["best_checkpoint"], self.head)
        self.assertEqual(best["epoch"], right["best_epoch"])
        self.assertEqual(best["history"], right["history"])
        for section in ("judgement", "gap_definitions", "gap_summary", "thresholds", "history_range"):
            self.assertEqual(continuous["diagnostics"][section], resumed["diagnostics"][section])
        for artifact in ("epoch_metrics_csv", "class_metrics_by_epoch_csv", "non_target_predictions_by_epoch_csv"):
            self.assertEqual(
                Path(continuous["diagnostics"]["artifacts"][artifact]).read_text(encoding="utf-8"),
                Path(resumed["diagnostics"]["artifacts"][artifact]).read_text(encoding="utf-8"),
            )
        bad = copy.deepcopy(right)
        bad["model_state_dict"]["proj.bias"][4] += 0.01
        path = self.root / "corrupt.pt"
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_official_checkpoint(path, self.head)
        bad = copy.deepcopy(right)
        next(iter(bad["optimizer_state_dict"]["state"].values()))["exp_avg"][4] = 1
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_official_checkpoint(path, self.head)
        bad = copy.deepcopy(right)
        bad["official_snapshot"]["revision"] = "other"
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_official_checkpoint(path, self.head)
        bad = copy.deepcopy(right)
        bad["checkpoint_id"] = "0" * 20
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_official_checkpoint(path, self.head)
        bad = copy.deepcopy(right)
        bad["checkpoint_id"] = "0" * 20
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_official_checkpoint(path, self.head)
        bad = copy.deepcopy(right)
        bad["best_validation_metrics"]["uar"] += 0.01
        torch.save(bad, path)
        with self.assertRaises(ValueError):
            load_official_checkpoint(path, self.head)
        artifact = self.artifacts["hcudb1"]
        result = evaluate_official(artifact.manifest_path, artifact.cache_root, "hcudb1", self.root / "eval",
                                  self.head, self.report, checkpoint_path=resumed["best_checkpoint"], device="cpu")
        self.assertEqual(len(result["result"]["predictions"][0]["logits9"]), 9)
        self.assertEqual(len(result["result"]["predictions"][0]["probabilities9"]), 9)
        self.assertEqual(np.asarray(result["result"]["metrics_target6"]["confusion_matrix_6x9"]).shape, (6, 9))

    def test_cache_and_configuration_rejections(self):
        artifact = self.artifacts["hcudb1"]
        store = ShardedFeatureStore(artifact.cache_root, artifact.manifest_path)
        bad = dict(store.meta, feature_dim=768)
        with self.assertRaises(ValueError):
            validate_official_cache(bad, self.head.provenance)
        with self.assertRaises(ValueError):
            validate_official_cache(store.meta, dict(self.head.provenance, revision="other"))
        with self.assertRaises(ValueError):
            train_official_decoder(artifact.manifest_path, artifact.cache_root, self.root / "bad", self.head, self.report,
                                   config=TrainingConfig(weight_decay=0.01))

    def test_full_train_console_artifacts_and_diagnostics_do_not_affect_training(self):
        artifact = self.artifacts["hcudb1"]
        config = TrainingConfig(epochs=1, weight_decay=0, batch_size=3, device="cpu")
        stream = io.StringIO()
        with redirect_stdout(stream):
            first = train_official_decoder(
                artifact.manifest_path, artifact.cache_root, self.root / "diagnostic-a",
                self.head, self.report, config=config,
                monitoring_config=TrainingMonitoringConfig(max_epoch_samples=1),
                diagnostics_config=OfficialTrainingDiagnosticsConfig(min_score_delta=0.01),
            )
        second = train_official_decoder(
            artifact.manifest_path, artifact.cache_root, self.root / "diagnostic-b",
            self.head, self.report, config=config,
            diagnostics_config=OfficialTrainingDiagnosticsConfig(min_score_delta=0.40),
        )
        output = stream.getvalue()
        for label in (
            "optimization train loss", "train loss", "validation loss", "train UAR",
            "validation UAR", "train macro F1", "validation macro F1", "train accuracy",
            "validation accuracy", "best epoch",
        ):
            self.assertIn(label, output)
        self.assertFalse(first["train_monitoring"]["is_subset"])
        self.assertEqual(first["train_monitoring"]["sample_size"], first["train_monitoring"]["population_size"])
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        left = load_official_checkpoint(first["resume_checkpoint"], self.head)
        right = load_official_checkpoint(second["resume_checkpoint"], self.head)
        for key, value in left["model_state_dict"].items():
            self.assertTrue(torch.equal(value, right["model_state_dict"][key]))
        for path in first["diagnostics"]["artifacts"].values():
            self.assertTrue(Path(path).is_file())

    def test_diagnostic_failure_occurs_after_resumable_checkpoint_save(self):
        artifact = self.artifacts["hcudb1"]
        output = self.root / "diagnostic-failure"
        with patch("ser_pipeline.diagnostics.write_official_training_diagnostics", side_effect=RuntimeError("plot failed")):
            with self.assertRaisesRegex(RuntimeError, "plot failed"):
                train_official_decoder(
                    artifact.manifest_path, artifact.cache_root, output,
                    self.head, self.report,
                    config=TrainingConfig(epochs=1, weight_decay=0, batch_size=3, device="cpu"),
                )
        last = output / "hcudb_official_continue_seed42_last.pt"
        self.assertTrue(last.is_file())
        self.assertEqual(load_official_checkpoint(last, self.head)["epoch"], 1)

    def test_three_seed_study_and_separate_final_comparison(self):
        with patch("ser_pipeline.training.evaluate_model", side_effect=AssertionError("test evaluation during training")):
            study = run_official_study(self.artifacts["hcudb1"], self.root / "study", self.head, self.report,
                                      config=TrainingConfig(epochs=1, weight_decay=0, device="cpu"))
        self.assertFalse(study["test_evaluated"])
        self.assertEqual(study["diagnostics_aggregate"]["status"], "complete")
        self.assertEqual(study["diagnostics_aggregate"]["completed_seed_count"], 3)
        self.assertTrue(all(run["diagnostics"]["path"] for run in study["runs"]))
        frozen = {run["seed"]: run["best"] for run in study["runs"]}
        final = run_official_final_evaluations(self.artifacts, frozen, self.root / "final", self.head, self.report, device="cpu")
        self.assertEqual(final["status"], "complete")
        self.assertEqual(len(final["evaluations"]), 8)
        self.assertEqual(sum(e["result"]["baseline"] for e in final["evaluations"]), 2)
        for dataset in self.artifacts:
            values = list(final["comparisons"][dataset]["uar"]["D_by_seed"].values())
            self.assertAlmostEqual(final["comparisons"][dataset]["uar"]["D_sample_std"], float(np.std(values, ddof=1)))
        wrong = copy.deepcopy(frozen)
        wrong[42]["sha256"] = "0" * 64
        with patch("ser_pipeline.training.evaluate_official", side_effect=AssertionError("evaluation before all identities fixed")):
            with self.assertRaises(ValueError):
                run_official_final_evaluations(self.artifacts, wrong, self.root / "badfinal", self.head, self.report, device="cpu")

    def test_notebook_defaults_and_cli_stages(self):
        from ser_pipeline.cli import build_parser
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/03_official_head_cd.ipynb").read_text())
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        namespace = {}
        with patch("ser_pipeline.study.run_official_study", side_effect=AssertionError("default training")), patch("ser_pipeline.study.run_official_final_evaluations", side_effect=AssertionError("default test evaluation")):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile(cell["source"], cell["id"], "exec"), namespace)
        self.assertTrue([key for key in namespace if key.startswith("RUN_")])
        self.assertTrue(all(not value for key, value in namespace.items() if key.startswith("RUN_")))
        self.assertEqual(namespace["CONFIG"].weight_decay, 0)
        self.assertEqual(namespace["CONFIG"].epochs, 10)
        self.assertEqual(namespace["DIAGNOSTICS_CONFIG"], OfficialTrainingDiagnosticsConfig())
        self.assertEqual(namespace["AUDIO_ROOTS"]["msp_podcast"].name, "MSP_PODCAST")
        self.assertEqual(namespace["AUDIO_ROOTS"]["hcudb1"].name, "HCUDB1")
        self.assertIsNone(namespace["MSP_EXPECTED_MISSING_SHA256"])
        self.assertEqual(namespace["MSP_APPROVED_DUPLICATE_EXCLUDE_IDS"], [])
        self.assertIsNone(namespace["MSP_EXPECTED_DUPLICATE_EXCLUSION_SHA256"])
        args = build_parser().parse_args(["study-official", "--snapshot", "s", "--parity-report", "p", "--manifest", "m", "--cache-root", "c", "--output-dir", "o"])
        self.assertEqual(args.seeds, [42, 43, 44])
        self.assertEqual(args.epochs, 10)


@unittest.skipUnless(os.environ.get("SER_OFFICIAL_SNAPSHOT"), "set SER_OFFICIAL_SNAPSHOT for real model parity")
class OfficialRealIntegrationTest(unittest.TestCase):
    def test_official_inference_and_disk_cache_parity(self):
        torch.set_num_threads(4)
        snapshot = Path(os.environ["SER_OFFICIAL_SNAPSHOT"])
        started = perf_counter()
        encoder = OfficialEmotion2vecEncoder(snapshot, device="cpu")
        waveform = load_audio_16k_mono(snapshot / "example" / "test.wav")
        report = encoder.verify_parity(waveform)
        before = state_sha256(encoder.model.state_dict())
        features = encoder.extract(waveform)
        c = OfficialHeadModel(encoder.head)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            np.save(path, features, allow_pickle=False)
            reloaded = torch.from_numpy(np.load(path, allow_pickle=False))
            torch.testing.assert_close(c(torch.from_numpy(features)[None]), c(reloaded[None]), rtol=1e-5, atol=1e-6)
        self.assertEqual(before, state_sha256(encoder.model.state_dict()))
        report["elapsed_seconds"] = perf_counter() - started
        report["mapped_tensor_count"] = len(encoder.key_map)
        output = os.environ.get("SER_OFFICIAL_PARITY_REPORT")
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
