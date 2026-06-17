import unittest

import torch
from torch import nn

from vad_downstream.model import Emotion2vecVADModel, VADRegressionHead


class DummyEmotion2vecEncoder(nn.Module):
    def __init__(self, features, padding_mask=None):
        super().__init__()
        self.probe = nn.Parameter(torch.ones(1))
        self.register_buffer("features", features)
        if padding_mask is None:
            padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        self.register_buffer("feature_padding_mask", padding_mask)
        self.last_source = None
        self.last_padding_mask = None

    def extract_features(
        self,
        source,
        padding_mask=None,
        mask=False,
        remove_extra_tokens=True,
    ):
        self.last_source = source
        self.last_padding_mask = padding_mask
        return {
            "x": self.features,
            "padding_mask": self.feature_padding_mask,
        }


class Emotion2vecVADModelTest(unittest.TestCase):
    def test_regression_head_returns_outputs_from_frame_features(self):
        head = VADRegressionHead(target_dim=2)
        features = torch.randn(2, 4, 768)
        padding_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, False],
            ]
        )

        output = head(features, padding_mask=padding_mask)

        self.assertEqual(tuple(output.shape), (2, 2))

    def test_va_model_returns_two_outputs_from_audio_input(self):
        encoder = DummyEmotion2vecEncoder(torch.randn(2, 4, 768))
        model = Emotion2vecVADModel(encoder=encoder, target_dim=2)
        source = torch.randn(2, 16000)

        output = model(source)

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertIs(encoder.last_source, source)

    def test_vad_model_returns_three_outputs_from_audio_input(self):
        encoder = DummyEmotion2vecEncoder(torch.randn(3, 5, 768))
        model = Emotion2vecVADModel(encoder=encoder, target_dim=3)
        source = torch.randn(3, 24000)

        output = model(source)

        self.assertEqual(tuple(output.shape), (3, 3))

    def test_pooling_ignores_padding_frames(self):
        features = torch.zeros(1, 3, 768)
        features[0, :, 0] = torch.tensor([1.0, 100.0, 3.0])
        padding_mask = torch.tensor([[False, True, False]])
        encoder = DummyEmotion2vecEncoder(features, padding_mask)
        model = Emotion2vecVADModel(encoder=encoder, target_dim=2, hidden_dim=1)

        with torch.no_grad():
            model.head.pre_net.weight.zero_()
            model.head.pre_net.weight[0, 0] = 1.0
            model.head.pre_net.bias.zero_()
            model.head.post_net.weight.fill_(1.0)
            model.head.post_net.bias.zero_()

        output = model(torch.randn(1, 16000))

        self.assertTrue(torch.allclose(output, torch.tensor([[2.0, 2.0]])))

    def test_rejects_fully_padded_feature_sequence(self):
        features = torch.randn(1, 2, 768)
        padding_mask = torch.tensor([[True, True]])
        encoder = DummyEmotion2vecEncoder(features, padding_mask)
        model = Emotion2vecVADModel(encoder=encoder, target_dim=2)

        with self.assertRaises(ValueError):
            model(torch.randn(1, 16000))

    def test_freezes_encoder_by_default_and_keeps_it_in_eval_mode(self):
        encoder = DummyEmotion2vecEncoder(torch.randn(1, 2, 768))
        model = Emotion2vecVADModel(encoder=encoder, target_dim=2)

        self.assertFalse(encoder.probe.requires_grad)
        self.assertFalse(encoder.training)

        model.train()

        self.assertTrue(model.training)
        self.assertFalse(encoder.training)


if __name__ == "__main__":
    unittest.main()
