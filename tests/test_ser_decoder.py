"""SER デコーダ、評価指標、チェックポイント互換性を検証する。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from iemocap_downstream.model import BaseModel as LegacyBaseModel
from ser_pipeline.audio import sha256_file
from ser_pipeline.checkpoints import (
    decoder_signature,
    load_decoder_checkpoint,
    restore_parent,
    restore_resume,
    save_decoder_checkpoint,
)
from ser_pipeline.evaluation import (
    assert_same_evaluation_sets,
    build_evaluation_result,
    classification_metrics,
    confusion_matrix,
    save_evaluation_result,
)
from ser_pipeline.model import BaseModel
from ser_pipeline.training import collate_features, selection_key


def cache_meta(feature_dim=4):
    return {
        "feature_layer": "final_after_encoder_norm",
        "feature_dim": feature_dim,
        "encoder_name": "fake",
        "encoder_checkpoint_sha256": "e" * 64,
    }


class SerDecoderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_legacy_state_dict_padding_and_all_padding_contract(self):
        self.assertIs(LegacyBaseModel, BaseModel)
        model = BaseModel(input_dim=4, output_dim=4, hidden_dim=3)
        plain = model(torch.ones(2, 3, 4))
        self.assertEqual(tuple(plain.shape), (2, 4))
        mask = torch.tensor([[False, False, True], [False, True, True]])
        padded = model(torch.ones(2, 3, 4), mask)
        self.assertEqual(tuple(padded.shape), (2, 4))
        with self.assertRaisesRegex(ValueError, "non-padding"):
            model(torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "feature dim"):
            model(torch.ones(1, 2, 5))

    def test_variable_length_collation(self):
        batch = collate_features(
            [
                (torch.ones(2, 4), 1, "a"),
                (torch.ones(4, 4), 2, "b"),
            ]
        )
        self.assertEqual(tuple(batch["net_input"]["feats"].shape), (2, 4, 4))
        self.assertEqual(batch["net_input"]["padding_mask"].tolist(), [[False, False, True, True], [False] * 4])
        self.assertEqual(batch["utterance_ids"], ["a", "b"])

    def test_metrics_probability_sum_iemocap_summary_and_saving(self):
        truth = [0, 1, 2, 3]
        prediction = [0, 2, 2, 1]
        probabilities = np.full((4, 4), 0.1, dtype=np.float64)
        probabilities[np.arange(4), prediction] = 0.7
        metrics = classification_metrics(truth, prediction, probabilities)
        self.assertEqual(metrics["accuracy"], metrics["wa"])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (4, 4))
        self.assertEqual(confusion_matrix(truth, prediction)[1, 2], 1)
        signature = {"dataset": "iemocap", "split": "test", "manifest_sha256": "a", "utterance_id_sha256": "b", "utterance_count": 4}
        result = build_evaluation_result(["a", "b", "c", "d"], truth, probabilities, dataset="iemocap", split="test", set_signature=signature)
        self.assertEqual(result["metrics_primary_3class"]["reported_class_indices"], [0, 1, 2])
        self.assertTrue(all(abs(sum(row["probabilities"]) - 1) < 1e-8 for row in result["predictions"]))
        paths = save_evaluation_result(result, self.root / "evaluation")
        self.assertTrue(all(Path(path).is_file() for path in paths.values()))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            classification_metrics(truth, prediction, np.ones((4, 4)))

    def test_selection_order_is_uar_then_macro_f1_then_loss(self):
        base = {"uar": 0.6, "macro_f1": 0.7, "loss": 0.5}
        self.assertGreater(selection_key({"uar": 0.7, "macro_f1": 0.1, "loss": 9}), selection_key(base))
        self.assertGreater(selection_key({"uar": 0.6, "macro_f1": 0.8, "loss": 9}), selection_key(base))
        self.assertGreater(selection_key({"uar": 0.6, "macro_f1": 0.7, "loss": 0.4}), selection_key(base))
        self.assertEqual(selection_key(base), selection_key(dict(base)))

    def test_probability_loss_clips_at_existing_floor_and_absent_classes_use_zero(self):
        probabilities = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.]])
        metrics = classification_metrics([0, 0], [1, 0], probabilities)
        self.assertEqual(metrics["loss"], -np.log(1e-12) / 2)
        self.assertEqual(metrics["uar"], .5 / 4)
        self.assertEqual(metrics["macro_f1"], (2 / 3) / 4)

    def test_parent_and_resume_are_distinct_and_strict(self):
        parent_model = BaseModel(input_dim=4, output_dim=4, hidden_dim=3)
        parent_optimizer = torch.optim.AdamW(parent_model.parameters(), lr=0.01)
        signature = decoder_signature(parent_model, 42, cache_meta())
        parent_path = self.root / "parent.pt"
        parent = save_decoder_checkpoint(
            parent_path,
            model=parent_model,
            optimizer=parent_optimizer,
            epoch=1,
            history=[{"epoch": 1}],
            training_stage="msp_train",
            signature=signature,
            run_id="run-parent",
            validation_metrics={"uar": 0.5, "macro_f1": 0.4, "loss": 1.0},
            cache_id="cache",
            mapping_versions=["msp"],
            split_versions=["split"],
        )
        child_model = BaseModel(input_dim=4, output_dim=4, hidden_dim=3)
        restored = restore_parent(child_model, parent_path, signature)
        self.assertEqual(restored["checkpoint_id"], parent["checkpoint_id"])
        child_optimizer = torch.optim.AdamW(child_model.parameters(), lr=0.02)
        child_path = self.root / "child.pt"
        child = save_decoder_checkpoint(
            child_path,
            model=child_model,
            optimizer=child_optimizer,
            epoch=1,
            history=[{"epoch": 1}],
            training_stage="hcudb_continue",
            signature=signature,
            run_id="run-child",
            validation_metrics={"uar": 0.4, "macro_f1": 0.3, "loss": 1.1},
            cache_id="cache2",
            mapping_versions=["hcudb"],
            split_versions=["split2"],
            parent_checkpoint=parent_path,
        )
        self.assertEqual(child["parent_checkpoint_id"], parent["checkpoint_id"])
        self.assertEqual(child["parent_checkpoint_sha256"], sha256_file(parent_path))

        resumed_model = BaseModel(input_dim=4, output_dim=4, hidden_dim=3)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.5)
        resumed = restore_resume(resumed_model, resumed_optimizer, child_path, signature, "hcudb_continue")
        self.assertEqual(resumed["run_id"], "run-child")
        bad_signature = dict(signature)
        bad_signature["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed"):
            load_decoder_checkpoint(parent_path, expected_signature=bad_signature)
        with self.assertRaisesRegex(ValueError, "training_stage"):
            restore_resume(resumed_model, resumed_optimizer, parent_path, signature, "hcudb_continue")

    def test_before_after_set_signatures_are_strict(self):
        signature = {"dataset": "msp", "split": "test", "manifest_sha256": "a", "utterance_id_sha256": "b", "utterance_count": 2}
        assert_same_evaluation_sets(signature, dict(signature))
        changed = dict(signature)
        changed["utterance_id_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "utterance_id_sha256"):
            assert_same_evaluation_sets(signature, changed)


if __name__ == "__main__":
    unittest.main()
