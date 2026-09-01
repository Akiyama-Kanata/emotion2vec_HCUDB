"""生成した SER ノートブック間で処理責務が混在していないことを検証する。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


class SerNotebookBoundaryTest(unittest.TestCase):
    def test_builder_check_is_write_free_and_ignores_line_endings(self):
        paths = [
            ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb",
            ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb",
            ROOT / "notebooks" / "msp_unavailable_label_audit.ipynb",
        ]
        before = [path.read_bytes() for path in paths]
        command = [sys.executable, str(ROOT / "scripts" / "build_ser_notebooks.py"), "--check"]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        self.assertEqual(before, [path.read_bytes() for path in paths])
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for path in paths[:2]:
                lf_bytes = path.read_bytes().replace(b"\r\n", b"\n")
                (output_dir / path.name).write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
            subprocess.run(
                command + ["--output-dir", str(output_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_feature_notebook_has_no_training_boundary_violations(self):
        notebook = json.loads((ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb").read_text(encoding="utf-8"))
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

    def test_decoder_notebook_is_cache_only_and_formal_run_is_disabled(self):
        notebook = json.loads((ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb").read_text(encoding="utf-8"))
        all_source = "\n".join(source_text(cell) for cell in notebook["cells"])
        code_source = "\n".join(source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("STUDY_DATASETS = ('msp_podcast', 'hcudb1')", code_source)
        self.assertIn("RUN_REAL_SMOKE = False", code_source)
        self.assertIn("RUN_FORMAL_SEED_42 = False", code_source)
        self.assertIn("RUN_FORMAL_SEEDS_43_44 = False", code_source)
        self.assertIn("FORMAL_EPOCHS = None", code_source)
        self.assertIn("CONFIRM_SMOKE_COMPLETED = False", code_source)
        self.assertIn("CONFIRM_SEED_42_ARTIFACTS = False", code_source)
        self.assertIn("require_formal_epochs(FORMAL_EPOCHS)", code_source)
        self.assertIn("device='cpu'", code_source)
        self.assertIn("ARTIFACT_DIR / 'smoke'", code_source)
        self.assertIn("ARTIFACT_DIR / 'formal' / 'initial-seed-42'", code_source)
        self.assertIn("ARTIFACT_DIR / 'formal' / 'followup-seeds-43-44'", code_source)
        self.assertIn("SER_MSP_PODCAST_EXCLUSION_CONTRACT", code_source)
        self.assertIn("SER_MSP_PODCAST_DUPLICATE_AUDIT", code_source)
        self.assertIn("SER_MSP_PODCAST_DUPLICATE_EXCLUSION_CONTRACT", code_source)
        self.assertIn("run_transfer_study", code_source)
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


if __name__ == "__main__":
    unittest.main()
