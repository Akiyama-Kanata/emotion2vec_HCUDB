import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vad_downstream import inference


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


class VADDownstreamInferenceTest(unittest.TestCase):
    def test_main_writes_va_json_with_random_head_and_dummy_encoder(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)
            output_path = Path(tmp_dir) / "prediction.json"

            payload = inference.main(
                [
                    "--wav",
                    str(wav_path),
                    "--target-dim",
                    "2",
                    "--allow-random-head",
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                ],
                encoder_factory=lambda args: DummyEmotion2vecEncoder(),
            )

            saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, saved_payload)
        self.assertEqual(saved_payload["labels"], ["valence", "arousal"])
        self.assertEqual(set(saved_payload["prediction"]), {"valence", "arousal"})
        self.assertIsNone(saved_payload["head_checkpoint"])
        self.assertTrue(saved_payload["random_head"])
        for value in saved_payload["prediction"].values():
            self.assertIsInstance(value, float)

    def test_main_writes_vad_json_with_random_head_and_dummy_encoder(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)
            output_path = Path(tmp_dir) / "prediction.json"

            payload = inference.main(
                [
                    "--wav",
                    str(wav_path),
                    "--target-dim",
                    "3",
                    "--allow-random-head",
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                ],
                encoder_factory=lambda args: DummyEmotion2vecEncoder(),
            )

        self.assertEqual(
            payload["labels"],
            ["valence", "arousal", "dominance"],
        )
        self.assertEqual(
            set(payload["prediction"]),
            {"valence", "arousal", "dominance"},
        )
        self.assertTrue(payload["random_head"])

    def test_rejects_missing_head_checkpoint_without_random_head_flag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)

            with self.assertRaisesRegex(ValueError, "--head-checkpoint is required"):
                inference.main(
                    [
                        "--wav",
                        str(wav_path),
                        "--target-dim",
                        "2",
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
