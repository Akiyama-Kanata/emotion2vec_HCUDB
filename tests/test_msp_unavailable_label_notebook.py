"""Verify the MSP unavailable-file label-audit notebook without real metadata."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "msp_unavailable_label_audit.ipynb"
BUILDER = ROOT / "scripts" / "build_ser_notebooks.py"


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def execute_code_cells() -> dict:
    namespace: dict = {}
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in load_notebook()["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = source_text(cell)
                exec(compile(source, f"{NOTEBOOK}#{cell['id']}", "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


class MspUnavailableLabelNotebookTest(unittest.TestCase):
    def test_builder_is_reproducible(self):
        before = NOTEBOOK.read_bytes()
        subprocess.run(
            [sys.executable, str(BUILDER)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        self.assertEqual(before, NOTEBOOK.read_bytes())

    def test_notebook_is_metadata_only_and_real_mode_is_disabled(self):
        notebook = load_notebook()
        code_source = "\n".join(
            source_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_REAL_DATA = False", code_source)
        self.assertIn("RUN_WAV_CSV_AUDIT = False", code_source)
        self.assertIn(r"C:\Users\RD004\Desktop\msp_unavailable_filenames.txt", code_source)
        self.assertIn("MSP_LABEL_CSV_PATH", code_source)
        self.assertIn("MSP_AUDIO_DIR", code_source)
        self.assertIn("labels_consensus.csv", code_source)
        self.assertIn("read_text", code_source)
        self.assertIn("read_csv", code_source)
        self.assertIn("scan_wav_directory", code_source)
        self.assertIn("audit_wav_csv", code_source)
        self.assertIn("map_emotion", code_source)
        for forbidden in (
            "import soundfile",
            "import librosa",
            "import torchaudio",
            "import wave",
            "wave.open",
            ".glob(",
            "os.walk(",
            ".iterdir(",
            ".to_csv(",
            ".to_excel(",
        ):
            self.assertNotIn(forbidden, code_source)

    def test_default_execution_uses_complete_synthetic_cases(self):
        namespace = execute_code_cells()
        self.assertFalse(namespace["RUN_REAL_DATA"])
        self.assertEqual("synthetic_only", namespace["data_mode"])

        summary = namespace["summary"].iloc[0]
        self.assertEqual(11, summary["raw_line_count"])
        self.assertEqual(1, summary["blank_line_count"])
        self.assertEqual(10, summary["nonempty_input_rows"])
        self.assertEqual(9, summary["unique_input_files"])
        self.assertEqual(1, summary["duplicate_input_rows"])
        self.assertEqual(8, summary["metadata_matched_files"])
        self.assertEqual(1, summary["metadata_unmatched_files"])
        self.assertEqual(5, summary["primary_4_files"])
        self.assertEqual(1, summary["outside_primary_4_files"])
        self.assertEqual(1, summary["empty_label_files"])
        self.assertEqual(1, summary["unknown_label_files"])

        statuses = set(namespace["details"]["mapping_status"])
        self.assertEqual(
            {
                "included_primary_4",
                "not_in_primary_4",
                "empty_label",
                "unknown_label",
                "metadata_not_found",
            },
            statuses,
        )
        mapped_counts = namespace["mapped_label_counts"].set_index("mapped_outcome")["count"]
        self.assertEqual(1, mapped_counts["anger"])
        self.assertEqual(2, mapped_counts["happy"])
        self.assertEqual(1, mapped_counts["sadness"])
        self.assertEqual(1, mapped_counts["disgust"])

    def test_text_and_csv_loaders_validate_metadata_edges(self):
        namespace = execute_code_cells()
        read_unavailable_list = namespace["read_unavailable_list"]
        read_msp_label_csv = namespace["read_msp_label_csv"]
        prepare_label_frame = namespace["prepare_label_frame"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "unavailable.txt"
            text_path.write_text("\ufeff X.wav \n\nx.WAV\nmissing.wav\n", encoding="utf-8")
            candidates, stats, duplicates = read_unavailable_list(text_path)
            self.assertEqual(4, stats["raw_line_count"])
            self.assertEqual(1, stats["blank_line_count"])
            self.assertEqual(2, stats["unique_input_files"])
            self.assertEqual(1, stats["duplicate_input_rows"])
            self.assertEqual(2, len(duplicates))
            self.assertEqual(["X.wav", "missing.wav"], candidates["input_filename"].tolist())

            csv_path = root / "labels.csv"
            csv_path.write_text("\ufeffFileName,EmoClass\nX.wav,A\n", encoding="utf-8")
            labels = read_msp_label_csv(csv_path)
            self.assertEqual(["A"], labels["original_emotion"].tolist())

            missing_column_path = root / "missing-column.csv"
            missing_column_path.write_text("\ufeffFileName\nX.wav\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                read_msp_label_csv(missing_column_path)

        duplicate_labels = pd.DataFrame(
            {"FileName": ["X.wav", "x.WAV"], "EmoClass": ["A", "H"]}
        )
        with self.assertRaisesRegex(ValueError, "duplicate normalized filenames"):
            prepare_label_frame(duplicate_labels)

    def test_wav_directory_and_csv_are_compared_without_audio_decoding(self):
        namespace = execute_code_cells()
        scan_wav_directory = namespace["scan_wav_directory"]
        prepare_label_frame = namespace["prepare_label_frame"]
        audit_wav_csv = namespace["audit_wav_csv"]

        with tempfile.TemporaryDirectory() as directory:
            audio_root = Path(directory) / "Audio"
            batch = audio_root / "batch_001"
            batch.mkdir(parents=True)
            (batch / "MSP-PODCAST_AVAILABLE.wav").touch()
            (batch / "MSP-PODCAST_UNLABELED.WAV").touch()

            labels = prepare_label_frame(
                pd.DataFrame(
                    {
                        "FileName": ["MSP-PODCAST_AVAILABLE.wav", "MSP-PODCAST_MISSING.wav"],
                        "EmoClass": ["A", "H"],
                    }
                )
            )
            wavs = scan_wav_directory(audio_root)
            summary, raw_counts, mapped_counts, _, missing_wavs, unlabeled_wavs = audit_wav_csv(
                wavs, labels
            )

            values = summary.iloc[0]
            self.assertEqual(2, values["wav_files_found"])
            self.assertEqual(2, values["label_csv_rows"])
            self.assertEqual(1, values["matched_wav_and_label"])
            self.assertEqual(1, values["label_rows_without_wav"])
            self.assertEqual(1, values["wav_files_without_label"])
            self.assertEqual(1, values["available_primary_4_files"])
            self.assertEqual({"A": 1}, raw_counts.set_index("original_emotion")["count"].to_dict())
            self.assertEqual(1, mapped_counts.set_index("mapped_emotion").loc["anger", "count"])
            self.assertEqual(["MSP-PODCAST_MISSING.wav"], missing_wavs["csv_filename"].tolist())
            self.assertEqual(["MSP-PODCAST_UNLABELED.WAV"], unlabeled_wavs["wav_filename"].tolist())

            second_batch = audio_root / "batch_002"
            second_batch.mkdir()
            (second_batch / "msp-podcast_available.WAV").touch()
            with self.assertRaisesRegex(ValueError, "duplicate normalized WAV filenames"):
                scan_wav_directory(audio_root)


if __name__ == "__main__":
    unittest.main()
