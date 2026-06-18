import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from vad_downstream import train_head


class VADDownstreamTrainHeadTest(unittest.TestCase):
    def test_cli_saves_stage3_head_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_prefix = self._write_dataset(
                tmp_dir,
                "train",
                lengths=[2, 2, 2],
                vad_rows=[
                    "utt0\t-0.5\t0.0",
                    "utt1\t0.0\t0.5",
                    "utt2\t0.5\t-0.5",
                ],
            )
            output_path = Path(tmp_dir) / "head.pt"

            with mock.patch("sys.stdout", new=io.StringIO()) as stdout:
                summary = train_head.main(
                    [
                        "--train-prefix",
                        train_prefix,
                        "--output",
                        str(output_path),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "3",
                        "--device",
                        "cpu",
                        "--seed",
                        "0",
                    ]
                )

            printed = json.loads(stdout.getvalue())
            self.assertTrue(output_path.exists())
            checkpoint = torch.load(output_path, map_location="cpu")

        self.assertEqual(summary, printed)
        self.assertIn("head_state_dict", checkpoint)
        self.assertEqual(checkpoint["target_dim"], 2)
        self.assertEqual(checkpoint["input_dim"], 768)
        self.assertEqual(checkpoint["hidden_dim"], 256)
        self.assertEqual(checkpoint["metadata"]["selection"], "final")
        self.assertEqual(summary["selection"], "final")
        self.assertEqual(summary["saved_epoch"], 1)

    def test_cli_rejects_train_valid_target_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_prefix = self._write_dataset(
                tmp_dir,
                "train",
                lengths=[2, 2],
                vad_rows=[
                    "utt0\t-0.5\t0.0",
                    "utt1\t0.5\t0.5",
                ],
            )
            valid_prefix = self._write_dataset(
                tmp_dir,
                "valid",
                lengths=[2, 2],
                vad_rows=[
                    "utt0\t-0.5\t0.0\t0.5",
                    "utt1\t0.5\t0.5\t-0.5",
                ],
            )
            output_path = Path(tmp_dir) / "head.pt"

            with self.assertRaisesRegex(ValueError, "target_dim"):
                train_head.main(
                    [
                        "--train-prefix",
                        train_prefix,
                        "--valid-prefix",
                        valid_prefix,
                        "--output",
                        str(output_path),
                        "--epochs",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )

    def _write_dataset(self, directory, name, lengths, vad_rows):
        prefix = Path(directory) / name
        total_frames = sum(lengths)
        values = np.linspace(
            -1.0,
            1.0,
            total_frames * 768,
            dtype=np.float32,
        ).reshape(total_frames, 768)
        np.save(str(prefix) + ".npy", values)

        with open(str(prefix) + ".lengths", "w", encoding="utf-8") as handle:
            for length in lengths:
                handle.write(f"{length}\n")

        with open(str(prefix) + ".vad", "w", encoding="utf-8") as handle:
            for row in vad_rows:
                handle.write(row + "\n")

        return str(prefix)


if __name__ == "__main__":
    unittest.main()
