"""Unit checks for dictionary-resolved official heads and protected adaptation."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from ser_pipeline.evaluation import official_classification_metrics, select_primary_logits
from ser_pipeline.model import OfficialHeadModel
from ser_pipeline.official import OfficialHead, OfficialLabelSpec, resolve_label_spec, strict_load_official_state
from ser_pipeline.training import OfficialRowGuard, train_one_epoch

LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")


def synthetic_head():
    torch.manual_seed(5)
    return OfficialHead(nn.Linear(1024, 9), OfficialLabelSpec(LABELS), {})


class OfficialUnitTest(unittest.TestCase):
    def test_d_updates_primary_and_preserves_other_rows_and_moments(self):
        head = synthetic_head()
        model = OfficialHeadModel(head, condition="D")
        guard = OfficialRowGuard(model, head)
        optimizer = torch.optim.AdamW(model.proj.parameters(), lr=0.001, weight_decay=0)
        guard.attach(optimizer)
        primary = list(head.label_spec.target_indices)
        before = copy.deepcopy(model.state_dict())
        for _ in range(5):
            optimizer.zero_grad()
            logits9 = model(torch.randn(8, 4, 1024))
            targets9 = torch.tensor(head.label_spec.target_indices)[torch.arange(8) % 6]
            loss = nn.functional.cross_entropy(logits9, targets9)
            loss.backward()
            for p in model.parameters():
                self.assertEqual(torch.count_nonzero(p.grad[list(guard.rows)]), 0)
            optimizer.step()
            guard.validate(optimizer)
        for key, value in model.state_dict().items():
            for index in primary:
                self.assertFalse(torch.equal(value[index], before[key][index]))
        optimizer.param_groups[0]["weight_decay"] = 0.01
        with self.assertRaises(ValueError):
            optimizer.step()
        optimizer.param_groups[0]["weight_decay"] = 0
        optimizer.state[model.proj.weight]["exp_avg"][guard.rows[0]] = 1
        with self.assertRaises(ValueError):
            guard.validate(optimizer)
        guard.restore(optimizer)
        guard.validate(optimizer)

    def test_training_rejects_c_and_unprotected_d(self):
        head = synthetic_head()
        for condition in ("C", "D"):
            model = OfficialHeadModel(head, condition=condition)
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0)
            with self.assertRaises(ValueError):
                train_one_epoch(model, optimizer, [], torch.device("cpu"))

    def test_dictionary_indices_and_conflicts(self):
        labels = tuple(reversed(LABELS))
        spec = resolve_label_spec(labels, {f"<{name}>": i for i, name in reversed(list(enumerate(labels)))})
        logits = torch.arange(9.)[None]
        expected = [labels.index(name) for name in ("angry", "happy", "sad", "disgusted")]
        self.assertEqual(select_primary_logits(logits, spec).tolist(), [[float(i) for i in expected]])
        for invalid in (LABELS[:-1], (*LABELS[:-1], "angry")):
            with self.assertRaises(ValueError):
                OfficialLabelSpec(invalid)
        with self.assertRaises(ValueError):
            resolve_label_spec(LABELS, labels)
        with self.assertRaises(ValueError):
            resolve_label_spec(LABELS, {name: 0 for name in LABELS})
        self.assertEqual(spec.target_names, ("angry", "disgusted", "fearful", "happy", "sad", "surprised"))
        self.assertEqual(spec.target_indices, tuple(labels.index(name) for name in spec.target_names))

    def test_primary_ignores_other_logits(self):
        spec = OfficialLabelSpec(LABELS)
        logits = torch.zeros(2, 9)
        expected = select_primary_logits(logits, spec).softmax(-1)
        logits[:, [i for i in range(9) if i not in spec.primary_indices]] = 1e8
        torch.testing.assert_close(select_primary_logits(logits, spec).softmax(-1), expected)
        torch.testing.assert_close(expected.sum(-1), torch.ones(2))
        with self.assertRaises(ValueError):
            select_primary_logits(logits)

    def test_official_metrics_keep_non_target_predictions(self):
        spec = OfficialLabelSpec(LABELS)
        predictions = [0, 4, 2, 5, 6, 8]
        probabilities = np.full((6, 9), 1e-6, dtype=np.float64)
        probabilities[np.arange(6), predictions] = 1.0 - 8e-6
        metrics = official_classification_metrics(range(6), probabilities, spec)
        self.assertEqual(np.asarray(metrics["confusion_matrix_6x9"]).shape, (6, 9))
        self.assertEqual(metrics["non_target_predictions"]["counts"], {"neutral": 1, "other": 1, "unknown": 1})
        self.assertEqual(metrics["non_target_predictions"]["total"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["uar"], 0.5)

    def test_c_fixed_independent_and_masked_pooling(self):
        head = synthetic_head()
        c, d = (OfficialHeadModel(head, condition=condition) for condition in ("C", "D"))
        before = copy.deepcopy(c.state_dict())
        x = torch.randn(2, 7, 1024)
        mask = torch.zeros(2, 7, dtype=torch.bool)
        mask[1, 3:] = True
        for _ in range(3):
            output = c(x, mask)
            self.assertEqual(output.shape, (2, 9))
            torch.testing.assert_close(output, d(x, mask), rtol=0, atol=0)
            torch.testing.assert_close(output[1], head.proj(x[1, :3].mean(0)), rtol=1e-5, atol=1e-6)
        self.assertEqual(c(x[:1]).shape, (1, 9))
        for key, tensor in c.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[key]))
        for cp, dp in zip(c.parameters(), d.parameters()):
            self.assertFalse(cp.requires_grad)
            self.assertNotEqual(cp.data_ptr(), dp.data_ptr())
        with self.assertRaises(ValueError):
            c.train()
        for invalid, padding in ((x[:, :0], None), (x[:, :, :768], None), (x, torch.ones_like(mask))):
            with self.assertRaises(ValueError):
                c(invalid, padding)

    def test_strict_load_checks_buffers_shapes_and_missing(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        source = {"encoder." + k: v.clone() for k, v in model.state_dict().items()}
        mapping = strict_load_official_state(model, source, ["encoder.", None])
        self.assertEqual(len(mapping), len(model.state_dict()))
        missing = dict(source)
        del missing["encoder.1.running_mean"]
        with self.assertRaises(ValueError):
            strict_load_official_state(model, missing, ["encoder.", None])
        source["encoder.0.weight"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            strict_load_official_state(model, source, ["encoder.", None])


if __name__ == "__main__":
    unittest.main()
