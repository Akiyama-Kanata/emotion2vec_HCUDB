import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from vad_downstream import inference
from vad_downstream.model import VADRegressionHead
from vad_downstream.training import save_head_checkpoint


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


class FakeFairseqEmotion2vec(nn.Module):
    def __init__(self, frame_count=4):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        features = torch.linspace(
            -0.5,
            0.5,
            frame_count * 768,
            dtype=torch.float32,
        ).view(1, frame_count, 768)
        self.register_buffer("features", features)
        self.to_calls = []
        self.extract_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def extract_features(
        self,
        source,
        padding_mask=None,
        mask=False,
        remove_extra_tokens=True,
    ):
        self.extract_calls.append(
            {
                "source": source.detach().cpu().clone(),
                "padding_mask": padding_mask,
                "mask": mask,
                "remove_extra_tokens": remove_extra_tokens,
            }
        )
        batch_size = source.size(0)
        features = self.features.to(source.device).expand(batch_size, -1, -1)
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

    def test_loads_stage3_head_checkpoint_without_random_head_flag(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)
            head_checkpoint = Path(tmp_dir) / "head.pt"
            output_path = Path(tmp_dir) / "prediction.json"
            save_head_checkpoint(
                VADRegressionHead(target_dim=2),
                head_checkpoint,
                target_dim=2,
            )

            payload = inference.main(
                [
                    "--wav",
                    str(wav_path),
                    "--target-dim",
                    "2",
                    "--head-checkpoint",
                    str(head_checkpoint),
                    "--output",
                    str(output_path),
                    "--device",
                    "cpu",
                ],
                encoder_factory=lambda args: DummyEmotion2vecEncoder(),
            )

        self.assertEqual(payload["labels"], ["valence", "arousal"])
        self.assertEqual(payload["head_checkpoint"], str(head_checkpoint))
        self.assertFalse(payload["random_head"])

    def test_rejects_stage3_head_checkpoint_target_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)
            head_checkpoint = Path(tmp_dir) / "head.pt"
            save_head_checkpoint(
                VADRegressionHead(target_dim=3),
                head_checkpoint,
                target_dim=3,
            )

            with self.assertRaisesRegex(ValueError, "target_dim .*does not match"):
                inference.main(
                    [
                        "--wav",
                        str(wav_path),
                        "--target-dim",
                        "2",
                        "--head-checkpoint",
                        str(head_checkpoint),
                        "--device",
                        "cpu",
                    ],
                    encoder_factory=lambda args: DummyEmotion2vecEncoder(),
                )

    def test_rejects_model_dir_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)

            with self.assertRaisesRegex(ValueError, "--checkpoint is required"):
                inference.main(
                    [
                        "--wav",
                        str(wav_path),
                        "--model-dir",
                        str(Path(tmp_dir) / "model"),
                        "--target-dim",
                        "2",
                        "--allow-random-head",
                        "--device",
                        "cpu",
                    ]
                )

    def test_rejects_checkpoint_without_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)

            with self.assertRaisesRegex(ValueError, "--model-dir is required"):
                inference.main(
                    [
                        "--wav",
                        str(wav_path),
                        "--checkpoint",
                        str(Path(tmp_dir) / "checkpoint.pt"),
                        "--target-dim",
                        "2",
                        "--allow-random-head",
                        "--device",
                        "cpu",
                    ]
                )

    def test_build_audio_encoder_uses_stage1_placeholder_without_checkpoint_args(self):
        encoder = inference.build_audio_encoder()

        self.assertIsInstance(encoder, inference.Stage1AudioFeatureEncoder)

    def test_main_uses_fake_fairseq_checkpoint_loader_and_writes_json(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = self._write_wav(tmp_dir)
            model_dir = Path(tmp_dir) / "emotion2vec"
            checkpoint = Path(tmp_dir) / "checkpoint.pt"
            output_path = Path(tmp_dir) / "prediction.json"
            fake_model = FakeFairseqEmotion2vec()
            import_user_modules = []
            checkpoint_loads = []

            def import_user_module(module):
                import_user_modules.append(module)

            def load_model_ensemble_and_task(paths):
                checkpoint_loads.append(paths)
                task = types.SimpleNamespace(
                    cfg=types.SimpleNamespace(normalize=True)
                )
                return [fake_model], types.SimpleNamespace(), task

            fake_fairseq = types.ModuleType("fairseq")
            fake_fairseq.utils = types.SimpleNamespace(
                import_user_module=import_user_module
            )
            fake_fairseq.checkpoint_utils = types.SimpleNamespace(
                load_model_ensemble_and_task=load_model_ensemble_and_task
            )

            with mock.patch.dict(sys.modules, {"fairseq": fake_fairseq}):
                payload = inference.main(
                    [
                        "--wav",
                        str(wav_path),
                        "--model-dir",
                        str(model_dir),
                        "--checkpoint",
                        str(checkpoint),
                        "--target-dim",
                        "2",
                        "--allow-random-head",
                        "--output",
                        str(output_path),
                        "--device",
                        "cpu",
                    ]
                )

            raw_source = inference.load_wav_16khz_mono(wav_path).unsqueeze(0)
            expected_source = torch.nn.functional.layer_norm(
                raw_source, raw_source.shape
            )
            saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, saved_payload)
        self.assertEqual(saved_payload["labels"], ["valence", "arousal"])
        self.assertEqual(set(saved_payload["prediction"]), {"valence", "arousal"})
        self.assertIsNone(saved_payload["head_checkpoint"])
        self.assertTrue(saved_payload["random_head"])
        self.assertEqual(len(import_user_modules), 1)
        self.assertEqual(import_user_modules[0].user_dir, str(model_dir))
        self.assertEqual(checkpoint_loads, [[str(checkpoint)]])
        self.assertFalse(fake_model.training)
        self.assertEqual(fake_model.to_calls[0][0], (torch.device("cpu"),))
        self.assertEqual(len(fake_model.extract_calls), 1)
        extract_call = fake_model.extract_calls[0]
        self.assertIsNone(extract_call["padding_mask"])
        self.assertFalse(extract_call["mask"])
        self.assertTrue(extract_call["remove_extra_tokens"])
        torch.testing.assert_close(extract_call["source"], expected_source)

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
