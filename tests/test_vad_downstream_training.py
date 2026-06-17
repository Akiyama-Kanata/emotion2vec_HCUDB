import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vad_downstream.data import VADSpeechDataset, load_vad_dataset
from vad_downstream.model import Emotion2vecVADModel, VADRegressionHead
from vad_downstream.training import (
    ccc_loss,
    concordance_correlation_coefficient,
    train_one_epoch,
)


class DummyEmotion2vecEncoder(nn.Module):
    def __init__(self, features, padding_mask=None):
        super().__init__()
        self.probe = nn.Parameter(torch.ones(1))
        self.register_buffer("features", features)
        if padding_mask is None:
            padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        self.register_buffer("feature_padding_mask", padding_mask)

    def extract_features(
        self,
        source,
        padding_mask=None,
        mask=False,
        remove_extra_tokens=True,
    ):
        return {
            "x": self.features,
            "padding_mask": self.feature_padding_mask,
        }


class VADDownstreamTrainingTest(unittest.TestCase):
    def test_ccc_and_loss_are_perfect_for_matching_predictions(self):
        prediction = torch.tensor(
            [
                [-1.0, 0.0],
                [0.0, 0.5],
                [1.0, 1.0],
            ]
        )

        ccc = concordance_correlation_coefficient(prediction, prediction)
        loss = ccc_loss(prediction, prediction)

        self.assertTrue(torch.allclose(ccc, torch.ones(2), atol=1e-6))
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_ccc_rejects_shape_mismatch_and_bad_target_dim(self):
        with self.assertRaises(ValueError):
            ccc_loss(torch.zeros(2, 2), torch.zeros(2, 3))

        with self.assertRaises(ValueError):
            ccc_loss(torch.zeros(2, 1), torch.zeros(2, 1))

    def test_train_one_epoch_updates_head_from_vad_dataset(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = self._write_dataset(tmp_dir)
            data = load_vad_dataset(prefix)
            dataset = VADSpeechDataset(
                data["feats"],
                data["sizes"],
                data["offsets"],
                data["targets"],
                data["utt_ids"],
            )
            loader = DataLoader(
                dataset,
                batch_size=3,
                collate_fn=dataset.collator,
                shuffle=False,
            )
            model = VADRegressionHead(target_dim=data["target_dim"], hidden_dim=8)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            before = model.post_net.weight.detach().clone()

            loss = train_one_epoch(model, optimizer, loader, torch.device("cpu"))

        self.assertIsInstance(loss, float)
        self.assertFalse(torch.allclose(model.post_net.weight, before))

    def test_train_one_epoch_accepts_source_input(self):
        torch.manual_seed(0)
        features = torch.randn(3, 4, 768)
        encoder = DummyEmotion2vecEncoder(features)
        model = Emotion2vecVADModel(encoder=encoder, target_dim=2, hidden_dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = {
            "net_input": {
                "source": torch.randn(3, 16000),
                "padding_mask": torch.zeros(3, 16000, dtype=torch.bool),
            },
            "target": torch.tensor(
                [
                    [-0.5, 0.0],
                    [0.0, 0.5],
                    [0.5, -0.5],
                ],
                dtype=torch.float32,
            ),
        }

        loss = train_one_epoch(
            model,
            optimizer,
            [batch],
            torch.device("cpu"),
            input_key="source",
        )

        self.assertIsInstance(loss, float)
        self.assertFalse(encoder.training)

    def test_train_one_epoch_rejects_empty_loader(self):
        model = VADRegressionHead(target_dim=2, hidden_dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with self.assertRaises(ValueError):
            train_one_epoch(model, optimizer, [], torch.device("cpu"))

    def _write_dataset(self, directory):
        prefix = Path(directory) / "sample"
        lengths = [2, 3, 1]
        total_frames = sum(lengths)
        features = np.linspace(
            -1.0,
            1.0,
            total_frames * 768,
            dtype=np.float32,
        ).reshape(total_frames, 768)
        np.save(str(prefix) + ".npy", features)

        with open(str(prefix) + ".lengths", "w", encoding="utf-8") as handle:
            for length in lengths:
                handle.write(f"{length}\n")

        with open(str(prefix) + ".vad", "w", encoding="utf-8") as handle:
            handle.write("utt0\t-0.5\t0.0\n")
            handle.write("utt1\t0.0\t0.5\n")
            handle.write("utt2\t0.5\t-0.5\n")

        return str(prefix)


if __name__ == "__main__":
    unittest.main()
