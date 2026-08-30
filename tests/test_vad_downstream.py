"""CSV 音声入力、欠損対応 CCC 損失、VAD 回帰モデルの旧現行経路を検証する。"""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vad_downstream"))

from data import attach_cache_paths, build_vad_dataloader, load_vad_csv, split_vad_records
from loss import vad_ccc_loss
from model import Emotion2VecVADRegressor, VAD_OUTPUT_NAMES
from train_vad import evaluate, train_one_epoch


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "vad_dummy"


class VADDownstreamTest(unittest.TestCase):
    def test_load_split_and_dataloader_from_dummy_fixture(self):
        records = load_vad_csv(str(FIXTURE_DIR / "vad_labels_dummy.csv"))
        records = attach_cache_paths(records, str(FIXTURE_DIR / "cache"))
        splits = split_vad_records(records, mode="split")

        self.assertEqual([len(splits[name]) for name in ("train", "val", "test")], [4, 2, 2])
        self.assertEqual(tuple(VAD_OUTPUT_NAMES), ("valence", "arousal", "dominance"))

        loader = build_vad_dataloader(splits["train"], batch_size=2)
        batch = next(iter(loader))

        self.assertEqual(batch["net_input"]["feats"].ndim, 3)
        self.assertEqual(tuple(batch["vad_labels"].shape), (2, 3))
        self.assertEqual(tuple(batch["vad_mask"].shape), (2, 3))
        self.assertTrue(torch.allclose(batch["vad_labels"][0], torch.tensor([0.50, 0.10, 0.30])))

    def test_vad_ccc_loss_uses_available_label_mask(self):
        pred = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=torch.float32)
        target = torch.tensor([[0.1, 0.0, 0.4], [0.6, 0.0, 0.8]], dtype=torch.float32)
        mask = torch.tensor([[True, False, True], [True, False, True]])

        loss = vad_ccc_loss(pred, target, mask)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.ndim, 0)

    def test_train_one_epoch_smoke_on_dummy_fixture(self):
        records = load_vad_csv(str(FIXTURE_DIR / "vad_labels_dummy.csv"))
        records = attach_cache_paths(records, str(FIXTURE_DIR / "cache"))
        splits = split_vad_records(records, mode="split")
        train_loader = build_vad_dataloader(splits["train"], batch_size=2, shuffle=True)
        val_loader = build_vad_dataloader(splits["val"], batch_size=2)

        model = Emotion2VecVADRegressor(input_dim=768, hidden_dim=16, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loss = train_one_epoch(model, train_loader, optimizer, torch.device("cpu"))
        metrics = evaluate(model, val_loader, torch.device("cpu"))

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["n_samples"], 2)


if __name__ == "__main__":
    unittest.main()
