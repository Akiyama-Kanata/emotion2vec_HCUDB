"""VAD経由カテゴリ感情分類の複合損失、評価指標、保存処理を検証する。"""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from vad_downstream.emotion_training import (
    compute_vad_emotion_loss,
    evaluate,
)
from vad_downstream.model import VADMediatedEmotionClassifier
from vad_downstream.training import ccc_loss


class FixedVADMediatedModel(nn.Module):
    def __init__(self, vad_prediction, logits):
        super().__init__()
        self.register_buffer("vad_prediction", vad_prediction)
        self.register_buffer("logits", logits)
        self.offset = 0

    def eval(self):
        self.offset = 0
        return super().eval()

    def forward(self, features, padding_mask=None, return_vad=False):
        batch_size = features.size(0)
        start = self.offset
        end = start + batch_size
        self.offset = end
        return {
            "vad": self.vad_prediction[start:end].to(features.device),
            "logits": self.logits[start:end].to(features.device),
        }


class VADDownstreamEmotionTrainingTest(unittest.TestCase):
    def test_combined_loss_matches_weighted_ccc_plus_cross_entropy(self):
        output = {
            "vad": torch.tensor(
                [
                    [-0.5, 0.0, 0.5],
                    [0.25, 0.5, -0.25],
                ],
                dtype=torch.float32,
            ),
            "logits": torch.tensor(
                [
                    [2.0, 0.0, -1.0, 0.5],
                    [0.0, 1.5, -0.5, 0.25],
                ],
                dtype=torch.float32,
            ),
        }
        vad_target = torch.tensor(
            [
                [-0.25, 0.0, 0.25],
                [0.5, 0.25, -0.5],
            ],
            dtype=torch.float32,
        )
        emotion_target = torch.tensor([0, 1], dtype=torch.long)

        losses = compute_vad_emotion_loss(
            output,
            vad_target,
            emotion_target,
            lambda_vad=0.5,
            lambda_emo=2.0,
        )

        expected_vad = ccc_loss(output["vad"], vad_target)
        expected_emo = F.cross_entropy(output["logits"], emotion_target)
        expected_total = 0.5 * expected_vad + 2.0 * expected_emo
        self.assertTrue(torch.allclose(losses["vad_loss"], expected_vad))
        self.assertTrue(torch.allclose(losses["emotion_loss"], expected_emo))
        self.assertTrue(torch.allclose(losses["loss"], expected_total))

    def test_combined_loss_backpropagates_to_vad_head_and_classifier(self):
        torch.manual_seed(0)
        model = VADMediatedEmotionClassifier(
            target_dim=3,
            num_classes=4,
            hidden_dim=8,
        )
        features = torch.randn(4, 3, 768)
        output = model(features, return_vad=True)
        vad_target = torch.tensor(
            [
                [-1.0, -0.5, 0.0],
                [-0.5, 0.0, 0.5],
                [0.5, 0.5, -0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        emotion_target = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        losses = compute_vad_emotion_loss(output, vad_target, emotion_target)
        losses["loss"].backward()

        self.assertIsNotNone(model.vad_head.post_net.weight.grad)
        self.assertIsNotNone(model.classifier.linear.weight.grad)
        self.assertGreater(float(model.vad_head.post_net.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.classifier.linear.weight.grad.abs().sum()), 0.0)

    def test_evaluate_returns_classification_and_vad_metrics(self):
        vad_target = torch.tensor(
            [
                [-1.0, -0.5, 0.0],
                [-0.5, 0.0, 0.5],
                [0.5, 0.5, -0.5],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        batches = [
            self._batch(vad_target[:2], target[:2]),
            self._batch(vad_target[2:], target[2:]),
        ]

        metrics = evaluate(
            FixedVADMediatedModel(vad_target, logits),
            batches,
            torch.device("cpu"),
            class_labels=["hap", "sad", "ang", "dis"],
        )

        self.assertAlmostEqual(metrics["valence_ccc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["arousal_ccc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["dominance_ccc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mean_ccc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["wa"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ua"], 1.0, places=6)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0, places=6)
        self.assertEqual(
            metrics["confusion_matrix"],
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        )

    def _batch(self, vad_target, emotion_target):
        return {
            "net_input": {
                "feats": torch.zeros(vad_target.size(0), 2, 768),
                "padding_mask": torch.zeros(vad_target.size(0), 2, dtype=torch.bool),
            },
            "vad_target": vad_target,
            "emotion_target": emotion_target,
        }


if __name__ == "__main__":
    unittest.main()
