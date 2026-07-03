import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vad_downstream import infer_vad_emotion
from vad_downstream.emotion_training import save_vad_emotion_checkpoint
from vad_downstream.model import VADMediatedEmotionClassifier


class DummyEmotion2vecEncoder(nn.Module):
    def __init__(self, frame_count=4):
        super().__init__()
        features = torch.linspace(
            -1.0,
            1.0,
            frame_count * 768,
            dtype=torch.float32,
        ).view(1, frame_count, 768)
        self.register_buffer("features", features)

    def extract_features(
        self,
        source,
        padding_mask=None,
        mask=False,
        remove_extra_tokens=True,
    ):
        batch_size = source.size(0)
        features = self.features.expand(batch_size, -1, -1)
        feature_padding_mask = torch.zeros(
            features.shape[:2], dtype=torch.bool, device=features.device
        )
        return {"x": features, "padding_mask": feature_padding_mask}


class VADDownstreamInferVADEmotionTest(unittest.TestCase):
    def test_main_writes_vad_emotion_json_with_explanations(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)
            classifier_checkpoint = Path(tmp_dir) / "vad_emotion.pt"
            output_path = Path(tmp_dir) / "prediction.json"
            model = VADMediatedEmotionClassifier(
                target_dim=3,
                num_classes=4,
                hidden_dim=8,
            )
            with torch.no_grad():
                model.classifier.linear.weight.copy_(
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.5],
                            [-0.5, 1.0, 0.0],
                            [0.0, -1.0, 0.5],
                            [-1.0, 0.5, -0.5],
                        ],
                        dtype=torch.float32,
                    )
                )
                model.classifier.linear.bias.copy_(
                    torch.tensor([0.1, -0.2, 0.3, 0.0], dtype=torch.float32)
                )
            save_vad_emotion_checkpoint(
                model,
                classifier_checkpoint,
                target_dim=3,
                class_labels=["hap", "sad", "ang", "dis"],
                class_names_ja=["喜び", "悲しみ", "怒り", "嫌悪"],
            )

            payload = infer_vad_emotion.main(
                [
                    "--wav",
                    str(wav_path),
                    "--classifier-checkpoint",
                    str(classifier_checkpoint),
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                ],
                encoder_factory=lambda args: DummyEmotion2vecEncoder(),
            )
            saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, saved_payload)
        self.assertEqual(payload["target_dim"], 3)
        self.assertEqual(payload["class_labels"], ["hap", "sad", "ang", "dis"])
        self.assertEqual(set(payload["vad"]), {"valence", "arousal", "dominance"})
        self.assertEqual(
            set(payload["linear_weights"]["hap"]),
            {"bias", "valence", "arousal", "dominance"},
        )
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0, places=6)
        for label in payload["class_labels"]:
            self.assertAlmostEqual(
                payload["contributions"][label]["logit_sum"],
                payload["logits"][label],
                places=5,
            )
        self.assertAlmostEqual(
            payload["contrast_to_runner_up"]["contributions"]["logit_sum"],
            payload["contrast_to_runner_up"]["logit_margin"],
            places=5,
        )

    def test_rejects_missing_classifier_checkpoint_without_random_flag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)

            with self.assertRaisesRegex(ValueError, "--classifier-checkpoint"):
                infer_vad_emotion.main(
                    [
                        "--wav",
                        str(wav_path),
                        "--device",
                        "cpu",
                    ],
                    encoder_factory=lambda args: DummyEmotion2vecEncoder(),
                )

    def _write_wav(self, directory):
        path = Path(directory) / "sample.wav"
        sample_rate = 16000
        samples = np.linspace(-0.25, 0.25, sample_rate, dtype=np.float32)
        pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")

        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

        return path


if __name__ == "__main__":
    unittest.main()
