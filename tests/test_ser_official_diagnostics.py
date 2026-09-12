"""Verify official6 history diagnostics, artifacts, and seed aggregation."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from ser_pipeline.diagnostics import (
    OfficialTrainingDiagnosticsConfig,
    aggregate_official_training_diagnostics,
    analyze_official_training_history,
    write_official_training_diagnostics,
)


TARGETS = ("angry", "disgusted", "fearful", "happy", "sad", "surprised")


def metrics(loss, uar, macro_f1, accuracy=0.5, support=2):
    class_metrics = [
        {
            "target_index": index,
            "target_label": label,
            "official_index": (0, 1, 2, 3, 6, 7)[index],
            "precision": 0.5,
            "recall": uar,
            "f1": macro_f1,
            "support": support,
        }
        for index, label in enumerate(TARGETS)
    ]
    return {
        "loss": loss,
        "uar": uar,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "non_target_predictions": {
            "counts": {"neutral": 1, "other": 2, "unknown": 3},
            "rates": {"neutral": 1 / 12, "other": 2 / 12, "unknown": 3 / 12},
            "total": 6,
            "rate": 0.5,
        },
    }


def history_row(epoch, train, validation, optimization_loss=0.9):
    return {
        "epoch": epoch,
        "train_loss": optimization_loss,
        "train_monitor": train,
        "validation": validation,
    }


FULL_TRAIN = {"is_subset": False, "sample_size": 12, "population_size": 12}


class OfficialTrainingDiagnosticsTest(unittest.TestCase):
    def test_gap_signs_and_overfitting_threshold_boundaries(self):
        history = [
            history_row(1, metrics(1.00, 0.50, 0.50), metrics(1.00, 0.50, 0.50)),
            history_row(2, metrics(0.98, 0.56, 0.55), metrics(1.01, 0.55, 0.54)),
            history_row(3, metrics(0.97, 0.62, 0.60), metrics(1.03, 0.60, 0.59)),
        ]
        result = analyze_official_training_history(
            history, best_epoch=2, train_monitoring=FULL_TRAIN,
        )
        self.assertEqual(result["judgement"]["status"], "overfitting")
        self.assertAlmostEqual(result["epoch_metrics"][-1]["uar_gap"], 0.02)
        self.assertAlmostEqual(result["epoch_metrics"][-1]["loss_gap"], 0.06)
        self.assertAlmostEqual(result["judgement"]["evidence"]["train_loss_decrease"], 0.03)
        self.assertAlmostEqual(result["judgement"]["evidence"]["validation_loss_increase"], 0.03)

    def test_underfitting_good_and_indeterminate_cases(self):
        underfit = [
            history_row(1, metrics(1.00, 0.38, 0.37), metrics(1.10, 0.37, 0.36)),
            history_row(2, metrics(0.99, 0.385, 0.375), metrics(1.09, 0.37, 0.36)),
            history_row(3, metrics(0.98, 0.39, 0.38), metrics(1.08, 0.38, 0.37)),
        ]
        result = analyze_official_training_history(underfit, best_epoch=3, train_monitoring=FULL_TRAIN)
        self.assertEqual(result["judgement"]["status"], "underfitting")

        threshold = copy.deepcopy(underfit)
        threshold[0]["train_monitor"]["uar"] = 0.49
        threshold[-1]["train_monitor"]["uar"] = 0.50
        self.assertEqual(
            analyze_official_training_history(threshold, best_epoch=3, train_monitoring=FULL_TRAIN)["judgement"]["status"],
            "good",
        )

        good = copy.deepcopy(underfit)
        good[-1]["train_monitor"]["uar"] = 0.60
        good[-1]["train_monitor"]["macro_f1"] = 0.60
        self.assertEqual(
            analyze_official_training_history(good, best_epoch=3, train_monitoring=FULL_TRAIN)["judgement"]["status"],
            "good",
        )

        cases = [
            (underfit[:2], FULL_TRAIN, "fewer_than_3_epochs"),
            (underfit, {"is_subset": True}, "partial_train_evaluation"),
        ]
        missing = copy.deepcopy(underfit)
        missing[-1]["validation"]["uar"] = float("nan")
        cases.append((missing, FULL_TRAIN, "missing_or_non_finite_metrics"))
        for history, monitoring, reason in cases:
            with self.subTest(reason=reason):
                diagnosed = analyze_official_training_history(
                    history, best_epoch=len(history), train_monitoring=monitoring,
                )
                self.assertEqual(diagnosed["judgement"]["status"], "indeterminate")
                self.assertIn(reason, diagnosed["judgement"]["reasons"])

    def test_csv_json_png_match_history_and_updates_leave_no_partial_files(self):
        history = [
            history_row(1, metrics(1.0, 0.4, 0.3), metrics(1.1, 0.35, 0.25)),
            history_row(2, metrics(0.9, 0.5, 0.4), metrics(1.0, 0.45, 0.35)),
            history_row(3, metrics(0.8, 0.6, 0.5), metrics(0.9, 0.55, 0.45)),
        ]
        original = copy.deepcopy(history)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = write_official_training_diagnostics(
                history[:1], output, run_id="run", seed=42, best_epoch=1,
                train_monitoring=FULL_TRAIN,
            )
            result = write_official_training_diagnostics(
                history, output, run_id="run", seed=42, best_epoch=2,
                train_monitoring=FULL_TRAIN,
            )
            self.assertEqual(history, original)
            self.assertEqual(first["diagnostics_schema_version"], "ser_official_training_diagnostics_v1")
            for path in result["artifacts"].values():
                self.assertTrue(Path(path).is_file())
            self.assertFalse(list(output.rglob("*.partial")))
            saved = json.loads(Path(result["artifacts"]["training_diagnostics_json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["history_range"]["epoch_count"], 3)

            with Path(result["artifacts"]["class_metrics_by_epoch_csv"]).open(encoding="utf-8", newline="") as source:
                class_rows = list(csv.DictReader(source))
            self.assertEqual(len(class_rows), 3 * 2 * 6)
            target = next(row for row in class_rows if row["epoch"] == "3" and row["split"] == "validation" and row["target_label"] == "fearful")
            self.assertEqual(float(target["recall"]), history[2]["validation"]["class_metrics"][2]["recall"])
            self.assertEqual(float(target["f1"]), history[2]["validation"]["class_metrics"][2]["f1"])
            self.assertEqual(int(target["support"]), history[2]["validation"]["class_metrics"][2]["support"])

            with Path(result["artifacts"]["non_target_predictions_by_epoch_csv"]).open(encoding="utf-8", newline="") as source:
                prediction_rows = list(csv.DictReader(source))
            unknown = next(row for row in prediction_rows if row["epoch"] == "2" and row["split"] == "train" and row["official_label"] == "unknown")
            self.assertEqual(int(unknown["count"]), 3)
            self.assertAlmostEqual(float(unknown["rate"]), 0.25)

    def test_three_seed_aggregation_uses_sample_standard_deviation_and_partial_status(self):
        diagnostics = []
        for index, value in enumerate((0.1, 0.2, 0.3)):
            gaps = {key: value for key in ("loss", "uar", "macro_f1", "accuracy")}
            diagnostics.append({
                "judgement": {"status": ("good", "good", "overfitting")[index]},
                "gap_summary": {"final": gaps, "best_epoch": dict(gaps)},
            })
        partial = aggregate_official_training_diagnostics(diagnostics[:2], requested_seed_count=3)
        self.assertEqual(partial["status"], "partial")
        complete = aggregate_official_training_diagnostics(diagnostics, requested_seed_count=3)
        self.assertEqual(complete["status"], "complete")
        self.assertAlmostEqual(complete["final_gaps"]["uar"]["mean"], 0.2)
        self.assertAlmostEqual(complete["final_gaps"]["uar"]["sample_std"], 0.1)
        self.assertEqual(complete["diagnosis_counts"], {
            "overfitting": 1, "underfitting": 0, "good": 2, "indeterminate": 0,
        })


if __name__ == "__main__":
    unittest.main()
