"""生成した SER ノートブック間で処理責務が混在していないことを検証する。"""

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


class SerNotebookBoundaryTest(unittest.TestCase):
    def test_builder_is_reproducible(self):
        paths = [
            ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb",
            ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb",
        ]
        before = [path.read_bytes() for path in paths]
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_ser_notebooks.py")], cwd=ROOT, check=True, capture_output=True)
        self.assertEqual(before, [path.read_bytes() for path in paths])

    def test_feature_notebook_has_no_training_boundary_violations(self):
        notebook = json.loads((ROOT / "notebooks" / "01_extract_emotion2vec_features.ipynb").read_text(encoding="utf-8"))
        code_source = "\n".join(source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("RUN_FULL_EXTRACTION = False", code_source)
        self.assertIn("audit_dataset", code_source)
        self.assertIn("mapping_summary", code_source)
        self.assertIn("split_summary", code_source)
        self.assertIn("demo_cache_summary", code_source)
        self.assertIn("one_item_feature_benchmark", code_source)
        for forbidden in ("optimizer", "train_decoder", "run_transfer_study", "BaseModel", "parent-checkpoint", "resume-checkpoint"):
            self.assertNotIn(forbidden, code_source)

    def test_decoder_notebook_is_cache_only_and_formal_run_is_disabled(self):
        notebook = json.loads((ROOT / "notebooks" / "02_train_and_evaluate_decoder.ipynb").read_text(encoding="utf-8"))
        all_source = "\n".join(source_text(cell) for cell in notebook["cells"])
        code_source = "\n".join(source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("RUN_FORMAL_STUDY = False", code_source)
        self.assertIn("STUDY_SEEDS = (42, 43, 44)", code_source)
        self.assertIn("run_transfer_study", code_source)
        self.assertIn("metrics['accuracy'] * 100", code_source)
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
