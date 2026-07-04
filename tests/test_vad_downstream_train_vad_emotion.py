import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from vad_downstream import train_vad_emotion


class VADDownstreamTrainVADEmotionTest(unittest.TestCase):
    def test_cli_saves_vad_mediated_classifier_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_prefix = self._write_dataset(tmp_dir, "train", target_dim=3)
            output_path = Path(tmp_dir) / "vad_emotion.pt"

            with mock.patch("sys.stdout", new=io.StringIO()) as stdout:
                summary = train_vad_emotion.main(
                    [
                        "--train-prefix",
                        train_prefix,
                        "--output",
                        str(output_path),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "4",
                        "--hidden-dim",
                        "8",
                        "--device",
                        "cpu",
                        "--seed",
                        "0",
                    ]
                )

            printed = json.loads(stdout.getvalue())
            checkpoint = torch.load(output_path, map_location="cpu")

            self.assertEqual(summary, printed)
            self.assertTrue(output_path.exists())
            self.assertIn("model_state_dict", checkpoint)
            self.assertEqual(checkpoint["target_dim"], 3)
            self.assertEqual(checkpoint["input_dim"], 768)
            self.assertEqual(checkpoint["hidden_dim"], 8)
            self.assertEqual(checkpoint["class_labels"], ["hap", "sad", "ang", "dis"])
            self.assertEqual(checkpoint["class_names_ja"], ["喜び", "悲しみ", "怒り", "嫌悪"])
            self.assertEqual(checkpoint["lambda_vad"], 1.0)
            self.assertEqual(checkpoint["lambda_emo"], 1.0)
            self.assertEqual(checkpoint["metadata"]["selection"], "final")
            self.assertEqual(summary["selection"], "final")
            self.assertEqual(summary["saved_epoch"], 1)
            self.assertEqual(summary["train"]["num_samples"], 4)

    def test_cli_rejects_target_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_prefix = self._write_dataset(tmp_dir, "train", target_dim=2)
            output_path = Path(tmp_dir) / "vad_emotion.pt"

            with self.assertRaisesRegex(ValueError, "target_dim"):
                train_vad_emotion.main(
                    [
                        "--train-prefix",
                        train_prefix,
                        "--output",
                        str(output_path),
                        "--epochs",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )

    def _write_dataset(self, directory, name, target_dim):
        prefix = Path(directory) / name
        lengths = [2, 2, 2, 2]
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

        vad_values = [
            ("utt0", [-0.5, 0.0, 0.5]),
            ("utt1", [0.0, 0.5, -0.5]),
            ("utt2", [0.5, -0.5, 0.0]),
            ("utt3", [1.0, 1.0, 1.0]),
        ]
        with open(str(prefix) + ".vad", "w", encoding="utf-8") as handle:
            for utt_id, values_for_utt in vad_values:
                kept = values_for_utt[:target_dim]
                handle.write(utt_id + "\t" + "\t".join(str(value) for value in kept) + "\n")

        with open(str(prefix) + ".emo", "w", encoding="utf-8") as handle:
            handle.write("utt0\thap\n")
            handle.write("utt1\tsad\n")
            handle.write("utt2\tang\n")
            handle.write("utt3\tdis\n")

        return str(prefix)


if __name__ == "__main__":
    unittest.main()
