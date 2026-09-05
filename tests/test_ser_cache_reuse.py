"""Non-training regressions for cache reuse, direct collation and study timing."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ser_pipeline import cache as cache_module
from ser_pipeline.audio import sha256_file
from ser_pipeline.cache import ShardedFeatureStore, _atomic_json
from ser_pipeline.checkpoints import decoder_signature, load_decoder_checkpoint, save_decoder_checkpoint
from ser_pipeline.evaluation import evaluation_set_signature
from ser_pipeline.model import BaseModel
from ser_pipeline.notebook_api import make_demo_artifacts
from ser_pipeline.study import FinalEvaluationTarget, prepare_study_stores, run_final_evaluations, run_transfer_study, summarize_study
from ser_pipeline.training import (
    CachedFeatureDataset, TrainingConfig, collate_features, evaluate_checkpoint, make_loader,
)


class SerCacheReuseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = make_demo_artifacts(self.root / "artifacts", datasets=("msp_podcast", "hcudb1"))
        self.artifact = self.artifacts["msp_podcast"]

    def tearDown(self):
        self.temporary.cleanup()

    def store(self, **kwargs):
        return ShardedFeatureStore(self.artifact.cache_root, self.artifact.manifest_path, **kwargs)

    def test_one_full_pass_and_same_mmaps_are_reused(self):
        with patch.object(cache_module, "_validate_shard_meta", wraps=cache_module._validate_shard_meta) as validate:
            store = self.store()
            self.assertEqual(validate.call_count, 3)
            feature = store.get(store.utterance_ids(split="train")[0])
            arrays = dict(store._arrays)
            for _ in range(12):
                store.ensure_validated()
            self.assertEqual(validate.call_count, 3)
            self.assertEqual(store.validation_count, 1)
            self.assertFalse(feature.flags.writeable)
            for path, array in arrays.items():
                self.assertIs(store._arrays[path], array)
        with patch.object(cache_module, "_validate_shard_meta", wraps=cache_module._validate_shard_meta) as validate:
            self.store(validate=False)
            self.assertEqual(validate.call_count, 3)

    def test_changed_input_revalidates_and_refreshes_maps(self):
        store = self.store()
        utterance_id = store.utterance_ids(split="train")[0]
        old_feature = store.get(utterance_id)
        old_arrays = dict(store._arrays)
        meta = self.artifact.cache_root / "cache_meta.json"
        meta.write_text(meta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        store.ensure_validated()
        self.assertEqual(store.validation_count, 2)
        self.assertFalse(store._arrays)
        np.testing.assert_array_equal(store.get(utterance_id), old_feature)
        for path, array in old_arrays.items():
            self.assertIsNot(store._arrays[path], array)
        with patch.object(cache_module.os, "getpid", return_value=os.getpid() + 1):
            store.ensure_validated()
        self.assertEqual(store.validation_count, 3)

    def test_every_input_kind_is_checked_before_reuse(self):
        root = self.artifact.cache_root
        paths = [
            self.artifact.manifest_path, root / "cache_meta.json",
            root / "msp_podcast/train/_SUCCESS",
            *sorted((root / "msp_podcast/train").glob("shard-*")),
        ]
        for path in paths:
            original = path.read_bytes()
            with self.subTest(path=path.name):
                store = self.store()
                store.get(store.utterance_ids(split="train")[0])
                # Windows does not allow replacing an open NumPy mmap file.
                store._arrays.clear()
                path.write_bytes(original + b"broken")
                try:
                    with self.assertRaises(ValueError):
                        store.ensure_validated()
                    self.assertFalse(store._arrays)
                    with self.assertRaisesRegex(ValueError, "successful validation"):
                        store.get("anything")
                finally:
                    path.write_bytes(original)

    def test_missing_or_added_files_and_validation_race_are_rejected(self):
        success = self.artifact.cache_root / "msp_podcast/train/_SUCCESS"
        original = success.read_bytes()
        store = self.store()
        success.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            store.ensure_validated()
        success.write_bytes(original)
        store.ensure_validated()
        added = success.with_name("shard-99999.meta.json")
        added.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing"):
            store.ensure_validated()
        added.unlink()
        real_validate = cache_module._validate_cache

        def changing_input(*args, **kwargs):
            result = real_validate(*args, **kwargs)
            meta = self.artifact.cache_root / "cache_meta.json"
            meta.write_text(meta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return result

        with patch.object(cache_module, "_validate_cache", side_effect=changing_input):
            with self.assertRaisesRegex(ValueError, "changed during"):
                store.ensure_validated()

    def test_nonfinite_values_are_rejected_even_with_matching_hashes(self):
        store = self.store()
        directory = self.artifact.cache_root / "msp_podcast/train"
        shard = directory / "shard-00000.npy"
        values = np.load(shard, allow_pickle=False)
        values[0, 0] = np.nan
        np.save(shard, values, allow_pickle=False)
        meta_path = directory / "shard-00000.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["shard_sha256"] = sha256_file(shard)
        _atomic_json(meta, meta_path)
        success = json.loads((directory / "_SUCCESS").read_text(encoding="utf-8"))
        success["shards"] = [meta]
        _atomic_json(success, directory / "_SUCCESS")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            store.ensure_validated()

    def test_payload_replacement_with_same_size_and_mtime_is_rejected(self):
        store = self.store()
        shard = self.artifact.cache_root / "msp_podcast/train/shard-00000.npy"
        stat = shard.stat()
        replacement = shard.with_name("replacement.npy")
        values = np.load(shard, allow_pickle=False)
        values[0, 0] += 1.0
        np.save(replacement, values, allow_pickle=False)
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(shard)
        self.assertEqual(shard.stat().st_size, stat.st_size)
        self.assertEqual(shard.stat().st_mtime_ns, stat.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            store.ensure_validated()

    def test_direct_batches_match_legacy_values_labels_masks_and_seed_order(self):
        store = self.store()
        for seed in (42, 43, 44):
            for split in ("train", "validation", "test"):
                with self.subTest(seed=seed, split=split):
                    dataset = CachedFeatureDataset(store, "msp_podcast", split)
                    sample = dataset[0][0]
                    self.assertTrue(np.shares_memory(sample, store.get(dataset.ids[0])))
                    actual = make_loader(store, "msp_podcast", split, batch_size=3, shuffle=True, seed=seed)
                    # Reproduce the original dataset/collator with owned tensors.
                    legacy = [
                        (torch.from_numpy(store.get(uid).copy()).float(), int(store.records[uid]["class_index"]), uid)
                        for uid in dataset.ids
                    ]
                    expected = torch.utils.data.DataLoader(
                        legacy, batch_size=3, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate_features,
                    )
                    for got, wanted in zip(actual, expected):
                        self.assertEqual(got["utterance_ids"], wanted["utterance_ids"])
                        self.assertTrue(torch.equal(got["labels"], wanted["labels"]))
                        for key in ("feats", "padding_mask"):
                            self.assertTrue(torch.equal(got["net_input"][key], wanted["net_input"][key]))
                        first_id = got["utterance_ids"][0]
                        original = store.get(first_id).copy()
                        got["net_input"]["feats"].fill_(99)
                        np.testing.assert_array_equal(store.get(first_id), original)

    def untrained_checkpoint(self, store, path, config, stage, parent=None):
        """Save initialized weights to test I/O contracts without optimization."""
        torch.manual_seed(config.seed)
        model = BaseModel(input_dim=8, hidden_dim=8)
        metrics = {"uar": 0.25, "macro_f1": 0.25, "wa": 0.25, "loss": 1.0}
        save_decoder_checkpoint(
            path, model=model, optimizer=torch.optim.AdamW(model.parameters()), epoch=1,
            history=[], training_stage=stage, signature=decoder_signature(model, config.seed, store.meta),
            run_id=f"io-only-{stage}-{config.seed}", validation_metrics=metrics,
            cache_id=store.meta["cache_id"], mapping_versions=store.meta["mapping_versions"],
            split_versions=store.meta["split_versions"], parent_checkpoint=parent,
            selection="best_validation", best_epoch=1, best_validation_metrics=metrics,
        )
        return {
            "best_checkpoint": str(path), "best_epoch": 1,
            "best_validation_metrics": metrics, "config": {"seed": config.seed},
        }

    def test_checkpoint_evaluation_matches_standalone_and_saves_predictions(self):
        store = self.store()
        checkpoint = self.root / "untrained.pt"
        self.untrained_checkpoint(store, checkpoint, TrainingConfig(), "msp_train")
        signature = evaluation_set_signature(self.artifact.manifest_path, "msp_podcast")
        with redirect_stdout(io.StringIO()):
            shared = evaluate_checkpoint(
                checkpoint, self.artifact.manifest_path, self.artifact.cache_root, "msp_podcast",
                self.root / "shared", device="cpu", batch_size=3, store=store,
            )
            standalone = evaluate_checkpoint(
                checkpoint, self.artifact.manifest_path, self.artifact.cache_root, "msp_podcast",
                self.root / "standalone", device="cpu", batch_size=3,
            )
        self.assertEqual(shared["result"], standalone["result"])
        self.assertEqual(shared["result"]["set_signature"], signature)
        self.assertEqual(shared["timings"]["cache_validation_seconds"], 0.0)
        self.assertGreater(standalone["timings"]["cache_validation_seconds"], 0.0)
        self.assertTrue(all(value >= 0 for value in shared["timings"].values()))
        self.assertTrue(all(Path(path).is_file() for path in shared["paths"].values()))
        self.assertEqual(
            Path(shared["paths"]["predictions"]).read_bytes(),
            Path(standalone["paths"]["predictions"]).read_bytes(),
        )
        other = self.artifacts["hcudb1"]
        with self.assertRaisesRegex(ValueError, "paths do not match"):
            evaluate_checkpoint(checkpoint, other.manifest_path, other.cache_root, "hcudb1", self.root / "wrong", store=store)

    def test_two_seed_study_reuses_gate_stores_without_training(self):
        def checkpoint_only(manifest, cache_root, dataset, output, config, *, training_stage, store, parent_checkpoint=None):
            self.assertIs(store, stores[dataset])
            store.require_paths(cache_root, manifest)
            store.ensure_validated()
            return self.untrained_checkpoint(store, output / "initialized.pt", config, training_stage, parent_checkpoint)

        with redirect_stdout(io.StringIO()), patch.object(
            cache_module, "_validate_shard_meta", wraps=cache_module._validate_shard_meta,
        ) as validate:
            stores = prepare_study_stores(self.artifacts)
            with patch("ser_pipeline.study.train_decoder", side_effect=checkpoint_only) as train, patch(
                "ser_pipeline.study.evaluate_checkpoint", side_effect=AssertionError("training must not run test"),
            ):
                summary = run_transfer_study(
                    self.artifacts, self.root / "study", seeds=(43, 44),
                    base_config=TrainingConfig(device="cpu", epochs=10), stores=stores,
                )
            self.assertEqual(validate.call_count, 6)  # Three splits per dataset, once.
            self.assertEqual(train.call_count, 4)
            self.assertTrue(all(call.args[4].epochs == 10 for call in train.call_args_list))
        original = copy.deepcopy(summary)
        compact = summarize_study(summary)
        self.assertEqual(summary, original)
        historical = copy.deepcopy(summary)
        historical.pop("test_evaluated")
        for run in historical["runs"]:
            run["before"] = {"msp_podcast": {"result": {"metrics_4class": {"uar": .999}}}}
            run["after"] = copy.deepcopy(run["before"])
        old_compact = summarize_study(historical)
        self.assertTrue(old_compact["test_evaluated"])
        self.assertNotIn("before", old_compact["runs"][0])
        self.assertEqual(old_compact["runs"][0]["parent"], compact["runs"][0]["parent"])
        self.assertNotIn('"predictions": [', json.dumps(compact))
        self.assertEqual(compact["seeds"], [43, 44])
        self.assertFalse(summary["test_evaluated"])
        for run in summary["runs"]:
            self.assertNotIn("before", run)
            self.assertNotIn("after", run)
            for dataset, splits in run["provenance"]["training_sets"].items():
                self.assertEqual(set(splits), {"train", "validation"})
                for split, signature in splits.items():
                    self.assertEqual(signature, evaluation_set_signature(self.artifacts[dataset].manifest_path, dataset, split))
        self.assertTrue(Path(summary["summary_path"]).is_file())
        self.assertTrue((self.root / "study/study_timings.json").is_file())
        for name in stores:
            self.assertEqual(summary["timings"]["cache_validation"][name]["full_passes"], 1)

    def test_final_evaluation_records_plan_and_reuses_stores_without_training(self):
        with redirect_stdout(io.StringIO()):
            stores = prepare_study_stores(self.artifacts)
        parent = self.root / "parent_best.pt"
        child = self.root / "child_best.pt"
        self.untrained_checkpoint(stores["msp_podcast"], parent, TrainingConfig(), "msp_train")
        self.untrained_checkpoint(stores["hcudb1"], child, TrainingConfig(), "hcudb_continue", parent)
        targets = [FinalEvaluationTarget(stage, checkpoint, sha256_file(checkpoint), dataset)
                   for stage, checkpoint in (("parent", parent), ("child", child)) for dataset in self.artifacts]
        output = self.root / "final"
        original_evaluate = evaluate_checkpoint
        def evaluate(*args, **kwargs):
            plan = json.loads((output / "final_evaluation_plan.json").read_text())
            self.assertEqual(len(plan["targets"]), 4)
            self.assertEqual(plan["batch_size"], 8)
            self.assertEqual(kwargs["split"], "test")
            return original_evaluate(*args, **kwargs)
        with patch("ser_pipeline.study.train_decoder", side_effect=AssertionError("final must not train")), patch(
            "ser_pipeline.study.evaluate_checkpoint", side_effect=evaluate
        ) as scoring, redirect_stdout(io.StringIO()):
            summary = run_final_evaluations(self.artifacts, targets, output, device="cpu", stores=stores)
        self.assertEqual(scoring.call_count, 4)
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["test_evaluated"])
        self.assertFalse((output / "study_summary.json").exists())
        for record in summary["evaluations"]:
            self.assertEqual(record["result"]["set_signature"]["split"], "test")
            self.assertEqual(record["target"]["expected_sha256"], sha256_file(record["target"]["checkpoint_path"]))
        for dataset in self.artifacts:
            results = [record["result"]["set_signature"] for record in summary["evaluations"] if record["target"]["dataset"] == dataset]
            self.assertEqual(results[0], results[1])
            self.assertEqual(stores[dataset].validation_count, 1)
        # MSP-only evaluation does not require HCUDB inputs.
        with redirect_stdout(io.StringIO()), patch("ser_pipeline.study.train_decoder", side_effect=AssertionError("must not train")):
            single = run_final_evaluations({"msp_podcast": self.artifact}, targets[:1], self.root / "single", device="cpu", stores={"msp_podcast": stores["msp_podcast"]})
        for store in stores.values():
            store._arrays.clear()
        self.assertEqual(len(single["evaluations"]), 1)

    def test_final_preflight_rejects_all_invalid_targets_before_any_evaluation(self):
        store = self.store()
        checkpoint = self.root / "best.pt"
        self.untrained_checkpoint(store, checkpoint, TrainingConfig(), "msp_train")
        good = FinalEvaluationTarget("selected", checkpoint, sha256_file(checkpoint), "msp_podcast")
        original = checkpoint.read_bytes()
        bad_sets = [[], [replace(good, expected_sha256="0" * 64)], [replace(good, checkpoint_path=self.root / "missing.pt")],
                    [good, replace(good, checkpoint_path=self.root / "missing-second.pt")], [good, good]]
        for index, targets in enumerate(bad_sets):
            with self.subTest(index=index), patch("ser_pipeline.study.evaluate_checkpoint") as evaluate, patch("ser_pipeline.study.train_decoder") as train:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    run_final_evaluations(self.artifacts, targets, self.root / f"bad-{index}", device="cpu", stores={"msp_podcast": store})
                evaluate.assert_not_called()
                train.assert_not_called()
        for field in ("selection", "signature", "best_epoch"):
            with self.subTest(field=field):
                checkpoint.write_bytes(original)
                payload = load_decoder_checkpoint(checkpoint)
                if field == "signature":
                    payload["signature"]["encoder_signature"]["encoder_checkpoint_sha256"] = "wrong"
                elif field == "selection":
                    payload["selection"] = "last"
                else:
                    payload["best_epoch"] = 2
                torch.save(payload, checkpoint)
                changed = replace(good, expected_sha256=sha256_file(checkpoint))
                with patch("ser_pipeline.study.evaluate_checkpoint") as evaluate, self.assertRaises(ValueError):
                    run_final_evaluations(self.artifacts, [changed], self.root / field, device="cpu", stores={"msp_podcast": store})
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
