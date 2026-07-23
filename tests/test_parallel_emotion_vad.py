import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vad_downstream.data import VADEmotionSpeechDataset, load_vad_emotion_dataset
from vad_downstream.infer_parallel_emotion_vad import make_payload, run_inference
from vad_downstream.model import ParallelEmotionVADClassifier
from vad_downstream.parallel_training import (
    compute_parallel_loss,
    save_parallel_checkpoint,
)
from vad_downstream import train_parallel_emotion_vad


class FakeEmotion2vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("features", torch.randn(1, 3, 768))

    def extract_features(self, source, padding_mask=None, **kwargs):
        features = self.features.expand(source.size(0), -1, -1)
        return {
            "x": features,
            "padding_mask": torch.zeros(features.shape[:2], dtype=torch.bool),
        }


class ParallelEmotionVADTest(unittest.TestCase):
    def test_model_always_returns_three_vad_values_and_variable_classes(self):
        model = ParallelEmotionVADClassifier(num_classes=6, hidden_dim=8)
        output = model(
            torch.randn(2, 4, 768),
            torch.tensor([[False, False, True, True], [False] * 4]),
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 6))
        self.assertEqual(tuple(output["vad"].shape), (2, 3))

    def test_mixed_loader_builds_fixed_targets_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = self._write_dataset(directory, include_d=(False, True, False))
            data = load_vad_emotion_dataset(prefix, class_labels=["one", "two"])
            dataset = VADEmotionSpeechDataset(
                data["feats"], data["sizes"], data["offsets"],
                data["vad_targets"], data["emotion_targets"],
                class_labels=data["class_labels"],
                vad_target_masks=data["vad_target_masks"],
            )
            batch = dataset.collator([dataset[index] for index in range(3)])
        self.assertEqual(tuple(batch["vad_target"].shape), (3, 3))
        self.assertEqual(
            batch["vad_target_mask"].tolist(),
            [[True, True, False], [True, True, True], [True, True, False]],
        )
        self.assertEqual(data["vad_label_counts"].tolist(), [3, 3, 1])

    def test_no_dominance_optimizer_step_leaves_d_head_bitwise_unchanged(self):
        torch.manual_seed(0)
        model = ParallelEmotionVADClassifier(hidden_dim=8)
        before = {key: value.detach().clone() for key, value in model.dominance_head.state_dict().items()}
        optimizer = torch.optim.AdamW(model.task_parameters(False), lr=0.01)
        output = model(torch.randn(4, 2, 768))
        mask = torch.tensor([[True, True, False]] * 4)
        losses = compute_parallel_loss(
            output, torch.randn(4, 3), mask, torch.tensor([0, 1, 2, 3])
        )
        losses["loss"].backward()
        optimizer.step()
        for key, value in model.dominance_head.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_all_heads_update_with_dominance_labels(self):
        torch.manual_seed(1)
        model = ParallelEmotionVADClassifier(hidden_dim=8)
        before = {
            name: next(module.parameters()).detach().clone()
            for name, module in (
                ("emotion", model.emotion_head),
                ("v", model.valence_head),
                ("a", model.arousal_head),
                ("d", model.dominance_head),
            )
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(torch.randn(4, 3, 768))
        loss = compute_parallel_loss(
            output,
            torch.tensor([[-1.0, -0.8, -0.6], [-0.4, -0.2, 0.0], [0.2, 0.4, 0.6], [0.8, 1.0, 0.9]]),
            torch.ones(4, 3, dtype=torch.bool),
            torch.tensor([0, 1, 2, 3]),
        )["loss"]
        loss.backward()
        optimizer.step()
        for name, module in (
            ("emotion", model.emotion_head), ("v", model.valence_head),
            ("a", model.arousal_head), ("d", model.dominance_head),
        ):
            self.assertFalse(torch.equal(next(module.parameters()), before[name]))

    def test_single_dominance_label_skips_d_loss(self):
        model = ParallelEmotionVADClassifier(hidden_dim=8)
        output = model(torch.randn(3, 2, 768))
        losses = compute_parallel_loss(
            output,
            torch.randn(3, 3),
            torch.tensor([[True, True, True], [True, True, False], [True, True, False]]),
            torch.tensor([0, 1, 2]),
        )
        losses["loss"].backward()
        self.assertTrue(losses["dominance_loss_skipped"])
        gradient = next(model.dominance_head.parameters()).grad
        self.assertTrue(gradient is None or torch.count_nonzero(gradient) == 0)

    def test_only_masked_dominance_targets_affect_d_loss(self):
        model = ParallelEmotionVADClassifier(hidden_dim=8)
        output = model(torch.randn(4, 2, 768))
        mask = torch.tensor(
            [[True, True, True], [True, True, False], [True, True, True], [True, True, False]]
        )
        target = torch.randn(4, 3)
        changed = target.clone()
        changed[~mask[:, 2], 2] += 1000.0
        first = compute_parallel_loss(output, target, mask, torch.tensor([0, 1, 2, 3]))
        second = compute_parallel_loss(output, changed, mask, torch.tensor([0, 1, 2, 3]))
        self.assertTrue(
            torch.equal(
                first["dimension_losses"]["dominance"],
                second["dimension_losses"]["dominance"],
            )
        )

    def test_task_losses_do_not_cross_into_other_heads(self):
        model = ParallelEmotionVADClassifier(hidden_dim=8)
        output = model(torch.randn(4, 2, 768))
        nn.CrossEntropyLoss()(output["logits"], torch.tensor([0, 1, 2, 3])).backward()
        self.assertIsNotNone(next(model.emotion_head.parameters()).grad)
        for head in (model.valence_head, model.arousal_head, model.dominance_head):
            self.assertIsNone(next(head.parameters()).grad)

        model.zero_grad(set_to_none=True)
        output = model(torch.randn(4, 2, 768))
        output["vad"].sum().backward()
        self.assertIsNone(next(model.emotion_head.parameters()).grad)

    def test_checkpoint_statuses_and_fake_wav_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parallel.pt"
            wav_path = Path(directory) / "sample.wav"
            self._write_wav(wav_path)
            model = ParallelEmotionVADClassifier(num_classes=2, hidden_dim=8)
            save_parallel_checkpoint(
                model, checkpoint, ["one", "two"], [3, 3, 0], "untrained"
            )
            result = run_inference(
                wav_path, checkpoint, device="cpu", encoder=FakeEmotion2vec()
            )
        self.assertEqual(result["vad"]["dominance"]["status"], "untrained")
        self.assertIn("warning", result)
        self.assertEqual(set(result["logits"]), {"one", "two"})
        self.assertEqual(set(result["probabilities"]), {"one", "two"})

        for status in ("trained", "untrained", "retained_from_checkpoint"):
            config = {
                "class_labels": ["one", "two"],
                "class_names_ja": ["one", "two"],
                "dominance_status": status,
                "vad_label_counts": {},
                "supervised_dimensions": [],
            }
            payload = make_payload("x.wav", "x.pt", torch.tensor([1.0, 0.0]), torch.zeros(3), config)
            self.assertEqual(payload["vad"]["dominance"]["status"], status)
            self.assertEqual("warning" in payload, status == "untrained")

    def test_cli_retains_trained_d_head_during_va_only_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            full_prefix = self._write_dataset(directory, include_d=(True, True, True), name="full")
            va_prefix = self._write_dataset(directory, include_d=(False, False, False), name="va")
            trained_path = Path(directory) / "trained.pt"
            retained_path = Path(directory) / "retained.pt"
            common = ["--epochs", "1", "--batch-size", "3", "--hidden-dim", "8", "--device", "cpu"]
            train_parallel_emotion_vad.main(
                ["--train-prefix", full_prefix, "--output", str(trained_path), "--class-labels", "one", "two"] + common
            )
            train_parallel_emotion_vad.main(
                ["--train-prefix", va_prefix, "--output", str(retained_path), "--initial-checkpoint", str(trained_path)] + common
            )
            trained = torch.load(trained_path, map_location="cpu")
            retained = torch.load(retained_path, map_location="cpu")
        self.assertEqual(trained["dominance_status"], "trained")
        self.assertEqual(retained["dominance_status"], "retained_from_checkpoint")
        self.assertEqual(retained["vad_label_counts"]["dominance"], 0)
        for key, value in trained["model_state_dict"].items():
            if key.startswith("dominance_head."):
                self.assertTrue(torch.equal(value, retained["model_state_dict"][key]))

    def _write_dataset(self, directory, include_d, name="mixed"):
        prefix = Path(directory) / name
        np.save(str(prefix) + ".npy", np.zeros((3, 768), dtype=np.float32))
        Path(str(prefix) + ".lengths").write_text("1\n1\n1\n", encoding="utf-8")
        rows = []
        for index, has_d in enumerate(include_d):
            suffix = "\t0.3" if has_d else ""
            rows.append(f"utt{index}\t0.1\t0.2{suffix}")
        Path(str(prefix) + ".vad").write_text("\n".join(rows) + "\n", encoding="utf-8")
        Path(str(prefix) + ".emo").write_text(
            "utt0\tone\nutt1\ttwo\nutt2\tone\n", encoding="utf-8"
        )
        return str(prefix)

    def _write_wav(self, path):
        samples = np.zeros(1600, dtype="<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())


if __name__ == "__main__":
    unittest.main()
