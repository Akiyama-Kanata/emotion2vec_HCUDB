"""連結済み特徴量と VAD／カテゴリ感情ラベルの読込・整合性検査を検証する。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vad_downstream.data import (
    EMOTION_CLASS_LABELS,
    VADEmotionSpeechDataset,
    VADSpeechDataset,
    load_vad_dataset,
    load_vad_emotion_dataset,
)


class VADDownstreamDataTest(unittest.TestCase):
    def test_loads_va_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[2, 1],
                vad_rows=[
                    "utt0\t-0.5\t0.25",
                    "utt1\t1.0\t-1.0",
                ],
            )

            data = load_vad_dataset(prefix)

        self.assertEqual(data["target_dim"], 2)
        self.assertEqual(data["num"], 2)
        self.assertEqual(data["utt_ids"], ["utt0", "utt1"])
        np.testing.assert_array_equal(data["sizes"], np.asarray([2, 1]))
        np.testing.assert_array_equal(data["offsets"], np.asarray([0, 2]))
        np.testing.assert_allclose(
            data["targets"], np.asarray([[-0.5, 0.25], [1.0, -1.0]], dtype=np.float32)
        )

    def test_loads_vad_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1, 1],
                vad_rows=[
                    "utt0\t-0.5\t0.25\t0.0",
                    "utt1\t1.0\t-1.0\t0.5",
                ],
            )

            data = load_vad_dataset(prefix)

        self.assertEqual(data["target_dim"], 3)
        np.testing.assert_allclose(
            data["targets"],
            np.asarray([[-0.5, 0.25, 0.0], [1.0, -1.0, 0.5]], dtype=np.float32),
        )

    def test_filters_lengths_and_keeps_alignment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1, 3, 5],
                vad_rows=[
                    "utt0\t0.0\t0.0",
                    "utt1\t0.1\t0.2",
                    "utt2\t0.3\t0.4",
                ],
            )

            data = load_vad_dataset(prefix, min_length=2, max_length=3)

        self.assertEqual(data["num"], 1)
        self.assertEqual(data["utt_ids"], ["utt1"])
        np.testing.assert_array_equal(data["sizes"], np.asarray([3]))
        np.testing.assert_array_equal(data["offsets"], np.asarray([1]))
        np.testing.assert_allclose(
            data["targets"], np.asarray([[0.1, 0.2]], dtype=np.float32)
        )

    def test_collator_pads_features_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, features = self._write_dataset(
                tmp_dir,
                lengths=[2, 3],
                vad_rows=[
                    "utt0\t-0.5\t0.25",
                    "utt1\t1.0\t-1.0",
                ],
            )
            data = load_vad_dataset(prefix)
            dataset = VADSpeechDataset(
                data["feats"],
                data["sizes"],
                data["offsets"],
                data["targets"],
                data["utt_ids"],
            )

            batch = dataset.collator([dataset[0], dataset[1]])

        self.assertEqual(batch["id"].tolist(), [0, 1])
        self.assertEqual(batch["utt_id"], ["utt0", "utt1"])
        self.assertEqual(tuple(batch["net_input"]["feats"].shape), (2, 3, 768))
        self.assertEqual(tuple(batch["net_input"]["padding_mask"].shape), (2, 3))
        self.assertEqual(
            batch["net_input"]["padding_mask"].tolist(),
            [[False, False, True], [False, False, False]],
        )
        self.assertEqual(tuple(batch["target"].shape), (2, 2))
        self.assertTrue(
            torch.allclose(
                batch["net_input"]["feats"][0, :2],
                torch.from_numpy(features[:2]).float(),
            )
        )
        self.assertTrue(torch.all(batch["net_input"]["feats"][0, 2] == 0))
        self.assertTrue(
            torch.allclose(
                batch["target"],
                torch.tensor([[-0.5, 0.25], [1.0, -1.0]], dtype=torch.float32),
            )
        )

    def test_loads_aligned_vad_emotion_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[2, 1, 3, 1],
                vad_rows=[
                    "utt0\t-0.5\t0.25\t0.0",
                    "utt1\t1.0\t-1.0\t0.5",
                    "utt2\t0.0\t0.0\t-0.5",
                    "utt3\t0.25\t0.5\t1.0",
                ],
                emo_rows=[
                    "utt0\thap",
                    "utt1\tsad",
                    "utt2\tang",
                    "utt3\tdis",
                ],
            )

            data = load_vad_emotion_dataset(prefix)

        self.assertEqual(data["target_dim"], 3)
        self.assertEqual(data["class_labels"], EMOTION_CLASS_LABELS)
        self.assertEqual(data["utt_ids"], ["utt0", "utt1", "utt2", "utt3"])
        np.testing.assert_array_equal(
            data["emotion_targets"],
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        self.assertEqual(data["emotion_labels"], ["hap", "sad", "ang", "dis"])
        np.testing.assert_allclose(
            data["vad_targets"],
            np.asarray(
                [
                    [-0.5, 0.25, 0.0],
                    [1.0, -1.0, 0.5],
                    [0.0, 0.0, -0.5],
                    [0.25, 0.5, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_vad_emotion_collator_pads_features_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[2, 3],
                vad_rows=[
                    "utt0\t-0.5\t0.25\t0.0",
                    "utt1\t1.0\t-1.0\t0.5",
                ],
                emo_rows=[
                    "utt0\thap",
                    "utt1\tdis",
                ],
            )
            data = load_vad_emotion_dataset(prefix)
            dataset = VADEmotionSpeechDataset(
                data["feats"],
                data["sizes"],
                data["offsets"],
                data["vad_targets"],
                data["emotion_targets"],
                data["utt_ids"],
                data["emotion_labels"],
            )

            batch = dataset.collator([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["net_input"]["feats"].shape), (2, 3, 768))
        self.assertEqual(tuple(batch["vad_target"].shape), (2, 3))
        self.assertEqual(tuple(batch["emotion_target"].shape), (2,))
        self.assertEqual(batch["emotion_target"].tolist(), [0, 3])
        self.assertEqual(batch["emotion_label"], ["hap", "dis"])
        self.assertTrue(torch.allclose(batch["target"], batch["vad_target"]))

    def test_rejects_unknown_emotion_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1],
                vad_rows=["utt0\t0.0\t0.0\t0.0"],
                emo_rows=["utt0\tneu"],
            )

            with self.assertRaisesRegex(ValueError, "unknown emotion label"):
                load_vad_emotion_dataset(prefix)

    def test_rejects_vad_emotion_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1],
                vad_rows=["utt0\t0.0\t0.0\t0.0"],
                emo_rows=["other\thap"],
            )

            with self.assertRaisesRegex(ValueError, "utterance_id mismatch"):
                load_vad_emotion_dataset(prefix)

    def test_rejects_vad_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1, 1],
                vad_rows=["utt0\t0.0\t0.0"],
            )

            with self.assertRaises(ValueError):
                load_vad_dataset(prefix)

    def test_rejects_mixed_vad_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1, 1],
                vad_rows=[
                    "utt0\t0.0\t0.0",
                    "utt1\t0.0\t0.0\t0.0",
                ],
            )

            with self.assertRaises(ValueError):
                load_vad_dataset(prefix)

    def test_rejects_vad_values_outside_normalized_range(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1],
                vad_rows=["utt0\t1.2\t0.0"],
            )

            with self.assertRaises(ValueError):
                load_vad_dataset(prefix)

    def test_rejects_wrong_feature_dimension(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[1],
                vad_rows=["utt0\t0.0\t0.0"],
                feature_dim=767,
            )

            with self.assertRaises(ValueError):
                load_vad_dataset(prefix)

    def test_rejects_frame_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix, _ = self._write_dataset(
                tmp_dir,
                lengths=[2, 2],
                vad_rows=[
                    "utt0\t0.0\t0.0",
                    "utt1\t0.0\t0.0",
                ],
                total_frames=3,
            )

            with self.assertRaises(ValueError):
                load_vad_dataset(prefix)

    def _write_dataset(
        self,
        directory,
        lengths,
        vad_rows,
        emo_rows=None,
        feature_dim=768,
        total_frames=None,
    ):
        prefix = Path(directory) / "sample"
        if total_frames is None:
            total_frames = sum(lengths)

        features = np.arange(total_frames * feature_dim, dtype=np.float32).reshape(
            total_frames, feature_dim
        )
        np.save(str(prefix) + ".npy", features)

        with open(str(prefix) + ".lengths", "w", encoding="utf-8") as handle:
            for length in lengths:
                handle.write(f"{length}\n")

        with open(str(prefix) + ".vad", "w", encoding="utf-8") as handle:
            for row in vad_rows:
                handle.write(row + "\n")

        if emo_rows is not None:
            with open(str(prefix) + ".emo", "w", encoding="utf-8") as handle:
                for row in emo_rows:
                    handle.write(row + "\n")

        return str(prefix), features


if __name__ == "__main__":
    unittest.main()
