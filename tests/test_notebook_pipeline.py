"""音声からカテゴリ感情・VADを扱うノートブック用パイプラインを検証する。"""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from vad_downstream.model import ParallelEmotionVADClassifier
from vad_downstream.notebook_pipeline import (
    DOMINANCE_WARNING,
    ColumnConfig,
    FeatureCache,
    TrainMinMaxNormalizer,
    assert_no_speaker_leakage,
    demo_feature_extractor,
    load_and_validate_annotations,
    make_dataset,
    make_loader,
    predict_wav_folder,
    speaker_split,
)
from vad_downstream.parallel_training import save_parallel_checkpoint, train_one_epoch


class NotebookPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.audio = self.root / "audio"
        self.audio.mkdir()
        self.columns = ColumnConfig()

    def tearDown(self):
        self.temporary.cleanup()

    def write_wav(self, name, sample_rate=16000, channels=1):
        data = np.zeros((1600, channels), dtype=np.float32) if channels > 1 else np.zeros(1600, dtype=np.float32)
        sf.write(self.audio / name, data, sample_rate)

    def valid_rows(self):
        rows = []
        for speaker in range(5):
            for emotion, offset in (("joy", 0), ("calm", 1)):
                name = f"s{speaker}_{emotion}.wav"
                self.write_wav(name)
                rows.append({
                    "audio": name, "speaker": f"s{speaker}", "emotion": emotion,
                    "valence": speaker + offset, "arousal": 10 - speaker + offset,
                })
        return rows

    def write_csv(self, rows, name="labels.csv"):
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_auto_labels_speaker_split_and_train_only_normalization(self):
        frame, labels = load_and_validate_annotations(self.write_csv(self.valid_rows()), self.audio, self.columns)
        self.assertEqual(labels, ["calm", "joy"])
        split = speaker_split(frame, "speaker", "emotion", seed=7)
        assert_no_speaker_leakage(split, "speaker")
        train = split[split.split == "train"]
        normalizer = TrainMinMaxNormalizer.fit(train, {"valence": "valence", "arousal": "arousal", "dominance": None})
        self.assertEqual(normalizer.minimum["valence"], float(train.valence.min()))
        self.assertEqual(normalizer.maximum["arousal"], float(train.arousal.max()))
        restored = TrainMinMaxNormalizer.from_dict(normalizer.to_dict())
        value = float(train.valence.iloc[0])
        self.assertAlmostEqual(restored.inverse_value("valence", restored.transform_value("valence", value)), value)

    def test_missing_column_bad_vad_unregistered_and_unsupported_audio(self):
        row = {"audio": "x.wav", "speaker": "s", "emotion": "joy", "valence": 1, "arousal": 2}
        self.write_wav("x.wav")
        for removed, message in (("speaker", "必要な列"), ("arousal", "必要な列")):
            broken = dict(row); broken.pop(removed)
            with self.assertRaisesRegex(ValueError, message):
                load_and_validate_annotations(self.write_csv([broken], removed + ".csv"), self.audio, self.columns)
        bad = dict(row); bad["valence"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "不正なVAD値"):
            load_and_validate_annotations(self.write_csv([bad], "bad.csv"), self.audio, self.columns)
        missing = dict(row); missing["audio"] = "missing.wav"
        with self.assertRaisesRegex(ValueError, "音声ファイルがありません"):
            load_and_validate_annotations(self.write_csv([missing], "missing.csv"), self.audio, self.columns)
        unsupported = dict(row); unsupported["audio"] = "x.mp3"
        with self.assertRaisesRegex(ValueError, "未対応音声形式"):
            load_and_validate_annotations(self.write_csv([unsupported], "format.csv"), self.audio, self.columns)

    def test_rejects_bad_audio_and_detects_explicit_speaker_leakage(self):
        self.write_wav("stereo.wav", channels=2)
        rows = [{"audio": "stereo.wav", "speaker": "s", "emotion": "joy", "valence": 1, "arousal": 2}]
        with self.assertRaisesRegex(ValueError, "16kHzモノラル"):
            load_and_validate_annotations(self.write_csv(rows), self.audio, self.columns)
        (self.audio / "corrupt.wav").write_bytes(b"not a wave file")
        rows[0]["audio"] = "corrupt.wav"
        with self.assertRaisesRegex(ValueError, "破損した音声"):
            load_and_validate_annotations(self.write_csv(rows, "corrupt.csv"), self.audio, self.columns)
        leaked = pd.DataFrame({"speaker": ["same", "same"], "split": ["train", "test"]})
        with self.assertRaisesRegex(ValueError, "複数split"):
            assert_no_speaker_leakage(leaked, "speaker")

    def test_dominance_head_is_unchanged_and_folder_csv_has_warning(self):
        frame, labels = load_and_validate_annotations(self.write_csv(self.valid_rows()), self.audio, self.columns)
        split = speaker_split(frame, "speaker", "emotion", seed=3)
        normalizer = TrainMinMaxNormalizer.fit(split[split.split == "train"], {"valence": "valence", "arousal": "arousal", "dominance": None})
        split = normalizer.transform_frame(split, {"valence": "valence", "arousal": "arousal", "dominance": None})
        cache = FeatureCache(self.root / "cache", "fake", demo_feature_extractor)
        train_set = make_dataset(split[split.split == "train"], labels, cache, self.columns)
        loader = make_loader(train_set, batch_size=len(train_set), shuffle=False)
        model = ParallelEmotionVADClassifier(num_classes=len(labels), hidden_dim=8)
        before = {key: value.detach().clone() for key, value in model.dominance_head.state_dict().items()}
        optimizer = torch.optim.AdamW(model.task_parameters(include_dominance=False), lr=0.01)
        train_one_epoch(model, optimizer, loader, "cpu")
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.dominance_head.state_dict()[key]))

        infer_dir = self.root / "infer"; infer_dir.mkdir()
        self.write_wav("inference.wav")
        (self.audio / "inference.wav").replace(infer_dir / "inference.wav")
        csv_path = self.root / "predictions.csv"
        result = predict_wav_folder(infer_dir, model, labels, normalizer, cache, csv_path)
        loaded = pd.read_csv(csv_path)
        self.assertEqual(result.loc[0, "dominance_status"], "untrained")
        self.assertTrue(np.isfinite(result.loc[0, "dominance"]))
        self.assertEqual(result.loc[0, "dominance_scale_status"], "unavailable_normalized_value_repeated")
        self.assertEqual(loaded.loc[0, "warning"], DOMINANCE_WARNING)
        self.assertIn(result.loc[0, "predicted_emotion"], labels)
        self.assertEqual(len([name for name in result.columns if name.startswith("probability_")]), len(labels))

    def test_checkpoint_additions_are_optional_and_round_trip(self):
        model = ParallelEmotionVADClassifier(num_classes=2, hidden_dim=8)
        normalizer = TrainMinMaxNormalizer({"valence": 1, "arousal": 2, "dominance": 0}, {"valence": 5, "arousal": 6, "dominance": 0}, {"valence": True, "arousal": True, "dominance": False})
        path = self.root / "model.pt"
        save_parallel_checkpoint(
            model, path, ["a", "b"], [2, 2, 0], "untrained",
            column_config=asdict(self.columns), vad_normalization=normalizer.to_dict(),
            encoder_info={"id": "fake", "frozen": True}, training_history=[{"epoch": 1}],
        )
        payload = torch.load(path, map_location="cpu")
        self.assertEqual(payload["class_labels"], ["a", "b"])
        restored = TrainMinMaxNormalizer.from_dict(payload["vad_normalization"])
        self.assertAlmostEqual(restored.inverse_value("valence", 0), 3.0)
        # Old-style calls still produce all keys required by existing readers.
        old_path = self.root / "old-compatible.pt"
        old = save_parallel_checkpoint(model, old_path, ["a", "b"], [1, 1, 0], "untrained")
        self.assertNotIn("vad_normalization", old)
        model.load_state_dict(old["model_state_dict"])


if __name__ == "__main__":
    unittest.main()
