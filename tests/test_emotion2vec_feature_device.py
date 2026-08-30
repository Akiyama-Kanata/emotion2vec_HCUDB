"""emotion2vec 特徴抽出器が指定した CPU／GPU 上で動くことを検証する。"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "iemocap_downstream" / "scripts" / "emotion2vec_speech_features.py"


class _FakeModel:
    def __init__(self):
        self.moved_to = None

    def eval(self):
        return self

    def to(self, device):
        self.moved_to = torch.device(device)
        return self


class FeatureExtractorDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_fairseq = types.SimpleNamespace(
            utils=types.SimpleNamespace(import_user_module=lambda _: None),
            checkpoint_utils=types.SimpleNamespace(),
        )
        modules = {
            "fairseq": fake_fairseq,
            "npy_append_array": types.SimpleNamespace(NpyAppendArray=object),
            "soundfile": types.SimpleNamespace(),
            "tqdm": types.SimpleNamespace(),
        }
        cls.patcher = mock.patch.dict(sys.modules, modules)
        cls.patcher.start()
        spec = importlib.util.spec_from_file_location("feature_device_under_test", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_parser_accepts_device(self):
        args = self.module.get_parser().parse_args([
            "--data", "x", "--model", "x", "--split", "train",
            "--checkpoint", "x", "--save-dir", "x", "--device", "cpu",
        ])
        self.assertEqual(args.device, "cpu")

    def test_reader_moves_model_to_cpu(self):
        model = _FakeModel()
        task = types.SimpleNamespace(cfg=types.SimpleNamespace(normalize=False))
        self.module.fairseq.checkpoint_utils.load_model_ensemble_and_task = lambda _: ([model], None, task)
        reader = self.module.Emotion2vecFeatureReader("x", "x", 0, device="cpu")
        self.assertEqual(reader.device.type, "cpu")
        self.assertEqual(model.moved_to.type, "cpu")


if __name__ == "__main__":
    unittest.main()
