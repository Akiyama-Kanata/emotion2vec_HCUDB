"""生成した SER ノートブック間で処理責務が混在していないことを検証する。"""

import json
import io
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


class SerNotebookBoundaryTest(unittest.TestCase):
    def test_comparison_settings_refresh_already_imported_modules(self):
        probe = """
import json
from pathlib import Path
import ser_pipeline.study as study
import ser_pipeline.training as training
import ser_pipeline.checkpoints as checkpoints

# Simulate a kernel retaining modules from before the comparison feature.
del study.load_msp_comparison_baselines
del study.run_msp_loss_comparison
del training.training_loss_config
training.TrainingConfig = object
checkpoints.save_decoder_checkpoint = None
notebook = json.loads(Path('notebooks/02_train_and_evaluate_decoder.ipynb').read_text(encoding='utf-8'))
cell = next(cell for cell in notebook['cells'] if cell['id'] == 'msp-loss-settings')
source = cell['source']
namespace = {}
exec(''.join(source) if isinstance(source, list) else source, namespace)
assert callable(namespace['load_msp_comparison_baselines'])
assert callable(namespace['run_msp_loss_comparison'])
assert callable(namespace['training_loss_config'])
assert callable(namespace['display_training_history'])
assert callable(namespace['load_saved_summary'])
assert namespace['MSP_COMPARISON_CONFIG'].class_weighting == 'none'
assert training.save_decoder_checkpoint is checkpoints.save_decoder_checkpoint
assert callable(checkpoints.save_decoder_checkpoint)
assert study.train_decoder is training.train_decoder
print('stale-kernel-refresh-ok')
"""
        result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("stale-kernel-refresh-ok", result.stdout)

    def test_builder_check_is_write_free_and_ignores_line_endings(self):
        paths = [
            ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb",
            ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb",
            ROOT / "notebooks" / "msp_unavailable_label_audit.ipynb",
        ]
        before = [path.read_bytes() for path in paths]
        command = [sys.executable, str(ROOT / "scripts" / "build_ser_notebooks.py"), "--check"]
        # Check generated defaults in a temporary directory. The working
        # notebook can legitimately contain user settings and saved results.
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            subprocess.run(command[:-1] + ["--output-dir", str(output_dir)], cwd=ROOT, check=True, capture_output=True)
            generated_before = {}
            for path in paths[:2]:
                generated_path = output_dir / path.name
                lf_bytes = generated_path.read_bytes().replace(b"\r\n", b"\n")
                (output_dir / path.name).write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
                generated_before[path.name] = generated_path.read_bytes()
            subprocess.run(
                command + ["--output-dir", str(output_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            self.assertEqual(generated_before, {name: (output_dir / name).read_bytes() for name in generated_before})
            altered = output_dir / paths[1].name
            payload = json.loads(altered.read_text(encoding="utf-8"))
            payload["cells"][0]["source"] = ["unexpected edit"]
            altered.write_text(json.dumps(payload), encoding="utf-8")
            rejected = subprocess.run(command + ["--output-dir", str(output_dir)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_feature_notebook_has_no_training_boundary_violations(self):
        builder = runpy.run_path(str(ROOT / "scripts" / "build_ser_notebooks.py"))
        notebook = builder["notebook"](builder["feature_cells"])
        code_source = "\n".join(source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("STUDY_DATASETS = ('msp_podcast', 'hcudb1')", code_source)
        for flag in (
            "RUN_MSP_AUDIT = False",
            "RUN_MSP_GENERATE_EXCLUSION_CONTRACT = False",
            "RUN_MSP_VERIFY_EXCLUSION_CONTRACT = False",
            "RUN_MSP_DUPLICATE_AUDIT = False",
            "RUN_MSP_GENERATE_DUPLICATE_EXCLUSION_CONTRACT = False",
            "RUN_MSP_VERIFY_DUPLICATE_EXCLUSION_CONTRACT = False",
            "RUN_MSP_BUILD_MANIFEST = False",
            "RUN_MSP_VALIDATE_MANIFEST = False",
            "RUN_MSP_BENCHMARK = False",
            "RUN_MSP_CAPACITY_GATE = False",
            "RUN_FULL_EXTRACTION = False",
            "RUN_VALIDATE_CACHE = False",
            "CONFIRM_MANIFEST_VALIDATED = False",
            "CONFIRM_BENCHMARK_AND_CAPACITY = False",
        ):
            self.assertIn(flag, code_source)
        self.assertIn("audit_dataset", code_source)
        self.assertIn("generate_msp_missing_audio_exclusion_contract", code_source)
        self.assertIn("load_msp_missing_audio_exclusion_contract", code_source)
        self.assertIn("APPROVED_MSP_EXCLUSION_SHA256 = None", code_source)
        self.assertIn("MSP_APPROVED_DUPLICATE_EXCLUDE_IDS = [", code_source)
        self.assertIn("APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256 = None", code_source)
        self.assertIn("generate_msp_audio_duplicate_audit", code_source)
        self.assertIn("generate_msp_audio_duplicate_exclusion_contract", code_source)
        self.assertIn("load_msp_audio_duplicate_audit", code_source)
        self.assertIn("load_msp_audio_duplicate_exclusion_contract", code_source)
        self.assertIn("approved_exclusion_contract=EXCLUSION_CONTRACT_PATH", code_source)
        self.assertIn("expected_exclusion_sha256=APPROVED_MSP_EXCLUSION_SHA256", code_source)
        self.assertIn("duplicate_audit=DUPLICATE_AUDIT_PATH", code_source)
        self.assertIn(
            "approved_duplicate_exclusion_contract=DUPLICATE_EXCLUSION_CONTRACT_PATH",
            code_source,
        )
        self.assertIn(
            "expected_duplicate_exclusion_sha256=APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256",
            code_source,
        )
        self.assertIn("build_manifest", code_source)
        self.assertIn("validate_manifest", code_source)
        self.assertIn("missing_eligible_original_label_counts", code_source)
        self.assertIn("missing_eligible_source_split_counts", code_source)
        self.assertIn("missing_eligible_kind_counts", code_source)
        self.assertIn("missing_eligible_original_by_source_split_counts", code_source)
        self.assertIn("missing_label_summary", code_source)
        self.assertIn("missing_label_split_cross", code_source)
        self.assertIn("mapping_summary", code_source)
        self.assertIn("split_summary", code_source)
        self.assertIn("benchmark_audio_extraction", code_source)
        self.assertIn("estimate_full_extraction", code_source)
        self.assertIn("disk_capacity_gate", code_source)
        self.assertIn("Emotion2vecEncoder", code_source)
        self.assertIn("extract_feature_cache", code_source)
        self.assertIn("validate_cache", code_source)
        self.assertIn("device='cpu'", code_source)
        self.assertIn("layer='final'", code_source)
        self.assertIn("expected_dim=768", code_source)
        self.assertNotIn("IEMOCAP_ROOT", code_source)
        cell_ids = {cell["id"] for cell in notebook["cells"]}
        self.assertTrue(
            {
                "exclusion-generation",
                "exclusion-verification",
                "exclusion-approval",
                "manifest-build",
                "missing-distribution",
                "duplicate-audit",
                "duplicate-decision",
                "duplicate-contract-generation",
                "duplicate-contract-verification",
                "duplicate-approval",
            }.issubset(cell_ids)
        )
        for forbidden in ("optimizer", "train_decoder", "run_transfer_study", "BaseModel", "parent-checkpoint", "resume-checkpoint"):
            self.assertNotIn(forbidden, code_source)
        working = json.loads((ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb").read_text(encoding="utf-8"))
        working_source = "\n".join(source_text(cell) for cell in working["cells"] if cell["cell_type"] == "code")
        for forbidden in ("optimizer", "train_decoder", "run_transfer_study", "BaseModel", "parent-checkpoint", "resume-checkpoint"):
            self.assertNotIn(forbidden, working_source)

    def test_decoder_notebook_is_cache_only_and_formal_run_is_disabled(self):
        builder = runpy.run_path(str(ROOT / "scripts" / "build_ser_notebooks.py"))
        notebook = builder["notebook"](builder["decoder_cells"])
        working = json.loads((ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb").read_text(encoding="utf-8"))
        working_cells = {cell["id"]: cell for cell in working["cells"]}
        for cell in notebook["cells"]:
            if cell["id"] == "msp-loss-settings":
                # Working settings (run flag, seeds, paths) are user-editable.
                working_settings = source_text(working_cells[cell["id"]])
                refresh = source_text(cell).split("import importlib\n", 1)[1].split("from ser_pipeline.cache", 1)[0]
                self.assertIn("import importlib\n" + refresh, working_settings)
            elif cell["id"] in {"smoke-gate", "formal-seed-42-gate", "formal-followup-gate"} or cell["id"].startswith(("msp-loss-", "final-test-")):
                self.assertEqual(source_text(working_cells[cell["id"]]), source_text(cell))
        all_source = "\n".join(source_text(cell) for cell in notebook["cells"])
        code_source = "\n".join(source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("STUDY_DATASETS = ('msp_podcast', 'hcudb1')", code_source)
        self.assertIn("RUN_REAL_SMOKE = False", code_source)
        self.assertIn("RUN_FORMAL_SEED_42 = False", code_source)
        self.assertIn("RUN_FORMAL_SEEDS_43_44 = False", code_source)
        self.assertIn("FORMAL_EPOCHS = 10", code_source)
        self.assertIn("CONFIRM_SMOKE_COMPLETED = False", code_source)
        self.assertIn("CONFIRM_SEED_42_ARTIFACTS = False", code_source)
        self.assertIn("require_formal_epochs(FORMAL_EPOCHS)", code_source)
        self.assertIn("device='cpu'", code_source)
        self.assertIn("TRAINING_OUTPUT_DIR / 'smoke'", code_source)
        self.assertIn("TRAINING_OUTPUT_DIR / 'formal' / 'initial-seed-42'", code_source)
        self.assertIn("TRAINING_OUTPUT_DIR / 'formal' / 'followup-seeds-43-44'", code_source)
        self.assertIn("SER_MSP_PODCAST_EXCLUSION_CONTRACT", code_source)
        self.assertIn("SER_MSP_PODCAST_DUPLICATE_AUDIT", code_source)
        self.assertIn("SER_MSP_PODCAST_DUPLICATE_EXCLUSION_CONTRACT", code_source)
        self.assertIn("run_transfer_study", code_source)
        self.assertIn("prepare_study_stores(artifacts)", code_source)
        self.assertIn("stores=followup_stores", code_source)
        self.assertIn("summarize_study(formal_followup_summary)", code_source)
        self.assertIn("RUN_MSP_WEIGHTED_TRAINING = False", code_source)
        self.assertIn("MSP_COMPARISON_SEEDS = (42,)", code_source)
        self.assertIn("run_msp_loss_comparison", code_source)
        for cell_id in ("smoke-gate", "formal-seed-42-gate", "formal-followup-gate", "msp-loss-run"):
            self.assertIn("display_training_history", source_text(working_cells[cell_id]))
            self.assertIn("load_saved_summary", source_text(working_cells[cell_id]))
        self.assertNotIn("display_training_history", source_text(working_cells['msp-loss-results']))
        self.assertIn("RUN_FINAL_TEST = False", code_source)
        self.assertIn("CONFIRM_FINAL_SETTINGS = False", code_source)
        self.assertIn("FINAL_TARGETS = [", code_source)
        working_code = '\n'.join(source_text(cell) for cell in working['cells'] if cell['cell_type'] == 'code')
        for flag in ('RUN_REAL_SMOKE', 'RUN_FORMAL_SEED_42', 'RUN_FORMAL_SEEDS_43_44', 'RUN_MSP_WEIGHTED_TRAINING', 'RUN_FINAL_TEST'):
            self.assertNotIn(flag + ' = True', working_code)
        ids = [cell['id'] for cell in working['cells']]
        self.assertLess(ids.index('environment-heading'), ids.index('setup'))
        self.assertLess(ids.index('configuration-heading'), ids.index('training-settings'))
        self.assertNotIn('59c02cbd', ids)
        self.assertNotIn('ca3be634', ids)
        self.assertNotIn("run_demo_transfer_study", code_source)
        self.assertNotIn("train_decoder", code_source)
        for forbidden in (".wav", "soundfile", "fairseq", "EMOTION2VEC_CHECKPOINT", "EMOTION2VEC_USER_DIR", "extract_feature_cache"):
            self.assertNotIn(forbidden, all_source)
        isolated = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['fairseq']=None; sys.modules['soundfile']=None; "
                "import ser_pipeline.cache, ser_pipeline.training, ser_pipeline.study; print('cache-only-ok')",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("cache-only-ok", isolated.stdout)

    def test_disabled_learning_cells_read_summaries_without_training_evaluation_or_cache(self):
        from ser_pipeline.notebook_api import display_training_history, load_saved_summary
        from ser_pipeline.study import summarize_study
        import matplotlib
        matplotlib.use('Agg')

        cells = {cell['id']: source_text(cell) for cell in runpy.run_path(str(ROOT / 'scripts/build_ser_notebooks.py'))['decoder_cells']}
        metrics = {'uar': .51234567, 'macro_f1': .4321, 'wa': .6, 'loss': 1.2}
        training = {'dataset': 'msp_podcast', 'seed': 42, 'best_epoch': 1,
                    'best_checkpoint': 'missing_checkpoint.pt', 'best_validation_metrics': metrics,
                    'history': [{'epoch': 1, 'train_loss': 1.7, 'validation': metrics}]}
        summary = {'seeds': [42], 'runs': [{'seed': 42, 'parent': training, 'child': training}], 'test_evaluated': False}
        comparison = {'runs': [{'seed': 42, 'baseline': {'training': training}, 'weighted': training}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'study_summary.json'
            path.write_text(json.dumps(summary), encoding='utf-8')
            comparison_path = Path(directory) / 'comparison_summary.json'
            comparison_path.write_text(json.dumps(comparison), encoding='utf-8')
            originals = {item: item.read_bytes() for item in (path, comparison_path)}
            forbidden = Mock(side_effect=AssertionError('display-only path started computation'))
            namespace = dict(
                RUN_REAL_SMOKE=False, RUN_FORMAL_SEED_42=False, RUN_FORMAL_SEEDS_43_44=False, RUN_MSP_WEIGHTED_TRAINING=False,
                SAVED_SMOKE_SUMMARY=path, SAVED_SEED_42_SUMMARY=path, SAVED_FOLLOWUP_SUMMARY=path,
                MSP_SAVED_COMPARISON_SUMMARY=comparison_path, display_training_history=display_training_history,
                load_saved_summary=load_saved_summary, summarize_study=summarize_study,
                validate_execution_gates=forbidden, run_transfer_study=forbidden, run_msp_loss_comparison=forbidden,
                ShardedFeatureStore=forbidden, load_msp_comparison_baselines=forbidden,
            )
            with patch('IPython.display.display') as display, patch('ser_pipeline.training.evaluate_checkpoint', forbidden), patch('ser_pipeline.training.evaluate_loader_metrics', forbidden), redirect_stdout(io.StringIO()):
                for cid in ('smoke-gate', 'formal-seed-42-gate', 'formal-followup-gate', 'msp-loss-prepare', 'msp-loss-run'):
                    exec(cells[cid], namespace)
            forbidden.assert_not_called()
            self.assertEqual(display.call_count, 8)  # One static HTML output for each stage/condition.
            for call in display.call_args_list:
                text = call.args[0].data
                self.assertEqual(text.count('<img '), 2)
                self.assertIn('<details><summary>lossを確認', text)
            self.assertEqual(originals, {item: item.read_bytes() for item in originals})
            self.assertEqual(set(Path(directory).iterdir()), set(originals))

    def test_final_notebook_gate_requires_confirmation_and_explicit_targets(self):
        cells = {cell['id']: source_text(cell) for cell in runpy.run_path(str(ROOT / 'scripts/build_ser_notebooks.py'))['decoder_cells']}
        scoring = Mock(side_effect=AssertionError('should not evaluate'))
        namespace = {'RUN_FINAL_TEST': False, 'run_final_evaluations': scoring}
        with redirect_stdout(io.StringIO()):
            exec(cells['final-test-gate'], namespace)
        namespace.update(RUN_FINAL_TEST=True, CONFIRM_FINAL_SETTINGS=False)
        with self.assertRaises(RuntimeError):
            exec(cells['final-test-gate'], namespace)
        namespace.update(CONFIRM_FINAL_SETTINGS=True, FINAL_TARGETS=[])
        with self.assertRaises(ValueError):
            exec(cells['final-test-gate'], namespace)
        scoring.assert_not_called()


if __name__ == "__main__":
    unittest.main()
