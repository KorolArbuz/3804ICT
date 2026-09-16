import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.model_quality_v2.config import V2Configuration
from src.model_quality_v2.operating_point import (
    POINT_NAMES,
    f_beta_from_counts,
    guard_legacy_state,
    manual_sklearn_parity,
    pareto_frontiers,
    predict_at_threshold,
    select_predeclared_points,
    threshold_frontier,
    tradeoff_against_reference,
    validate_oof,
)


def selection_frontier():
    counts = [
        # threshold, TN, FP, FN, TP
        (.2, 5, 5, 2, 8),
        (.3, 7, 3, 3, 7),
        (.4, 9, 1, 5, 5),
        (.5, 10, 0, 6, 4),
    ]
    rows = []
    for threshold, tn, fp, fn, tp in counts:
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        rows.append({
            "threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": (tn + tp) / 20, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": 2 * tp / (2 * tp + fp + fn),
            "f0_5": float(f_beta_from_counts(tp, fp, fn)),
            "balanced_accuracy": (recall + specificity) / 2,
            "predicted_positive_rate": (tp + fp) / 20,
            "false_positive_rate": fp / (fp + tn), "false_negative_rate": fn / (fn + tp),
        })
    return pd.DataFrame(rows)


class ThresholdFrontierTests(unittest.TestCase):
    def test_greater_equal_threshold_semantics(self):
        np.testing.assert_array_equal(predict_at_threshold([.2, .3, .4], .3), [0, 1, 1])

    def test_exact_frontier_counts(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .4, .4, .8])
        result = threshold_frontier(labels, scores)
        at_point_four = result[result.threshold == .4].iloc[0]
        self.assertEqual((at_point_four.TN, at_point_four.FP,
                          at_point_four.FN, at_point_four.TP), (1, 1, 0, 2))
        self.assertTrue((result.threshold == np.nextafter(1.0, np.inf)).any())

    def test_f0_5_formula(self):
        expected = 1.25 * 8 / (1.25 * 8 + .25 * 2 + 5)
        self.assertAlmostEqual(float(f_beta_from_counts(8, 5, 2)), expected)


class OperatingPointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.points = select_predeclared_points(selection_frontier(), .2)

    def test_min_fp_recall_constraint(self):
        self.assertEqual(self.points["min_fp_recall_50"].threshold, .4)

    def test_min_fp_f1_constraint(self):
        self.assertEqual(self.points["min_fp_f1_52"].threshold, .5)

    def test_accuracy_guard(self):
        self.assertEqual(self.points["min_fp_accuracy_guard"].threshold, .4)

    def test_max_precision_with_recall_constraint(self):
        self.assertEqual(self.points["max_precision_recall_50"].threshold, .4)

    def test_f0_5_selection(self):
        self.assertEqual(self.points["f0_5_optimized"].threshold, .5)

    def test_impossible_constraint_is_explicit(self):
        frame = selection_frontier().copy()
        frame["recall"] = .4
        with self.assertRaisesRegex(ValueError, "min_fp_recall_50"):
            select_predeclared_points(frame, .2)

    def test_pareto_dominance(self):
        frame = selection_frontier()
        extra = frame.iloc[[2]].copy()
        extra["threshold"] = .45
        extra["FP"] = 2  # Dominated by threshold .4 with lower FP and equal recall/F1.
        combined = pd.concat([frame, extra], ignore_index=True)
        result, size_3d, size_2d = pareto_frontiers(combined)
        dominated = result[result.threshold == .45]
        self.assertTrue(dominated.empty or not dominated.pareto_3d.iloc[0])
        self.assertGreater(size_3d, 0)
        self.assertGreater(size_2d, 0)


class ProtocolGuardTests(unittest.TestCase):
    def _valid_oof(self):
        train = np.arange(24000)
        dataset = SimpleNamespace(
            row_ids=np.arange(30000), original_ids=np.arange(30000).astype(str),
            y=np.tile([0, 1], 15000),
        )
        frame = pd.DataFrame({
            "training_position": np.arange(24000), "row_id": train,
            "original_id": train.astype(str), "fold": np.tile(np.arange(1, 6), 4800),
            "y_true": dataset.y[train], "probability_class_1": np.full(24000, .5),
        })
        return frame, dataset, train

    def test_all_24000_oof_rows_predicted_once_without_test_rows(self):
        frame, dataset, train = self._valid_oof()
        validate_oof(frame, dataset, train)
        self.assertEqual(len(frame), 24000)
        self.assertFalse(frame.training_position.duplicated().any())

    def test_oof_test_row_leakage_is_rejected(self):
        frame, dataset, train = self._valid_oof()
        frame.loc[0, "row_id"] = 29999
        with self.assertRaises(ValueError):
            validate_oof(frame, dataset, train)

    def test_freeze_and_legacy_rerun_guards(self):
        guard_legacy_state({"legacy_test_evaluated": False})
        with self.assertRaises(ValueError):
            guard_legacy_state({"legacy_test_evaluated": True})
        guard_legacy_state({"legacy_test_evaluated": True}, verify_rerun=True)
        with self.assertRaises(ValueError):
            guard_legacy_state({"legacy_test_evaluated": False}, verify_rerun=True)

    def test_tradeoff_arithmetic(self):
        reference = {"FP": 10, "FN": 4, "TP": 6, "TN": 20,
                     "accuracy": .65, "precision": .375, "recall": .6, "f1": .46, "f0_5": .4}
        candidate = {"FP": 4, "FN": 6, "TP": 4, "TN": 26,
                     "accuracy": .75, "precision": .5, "recall": .4, "f1": .44, "f0_5": .48}
        result = tradeoff_against_reference(reference, candidate)
        self.assertEqual(result["FP_avoided"], 6)
        self.assertEqual(result["TP_lost"], 2)
        self.assertEqual(result["net_correct_classification_change"], 4)
        self.assertAlmostEqual(result["TP_lost_per_100_FP_avoided"], 100 / 3)

    def test_manual_sklearn_all_supplied_rows_path(self):
        X_train = np.array([[0.], [1.], [3.], [6.]])
        y_train = np.array([0, 1, 1, 0])
        X_test = np.array([[.2], [2.], [5.]])
        config = V2Configuration(k=3)
        sklearn_scores = KNeighborsClassifier(
            3, metric="euclidean", weights="distance", algorithm="brute"
        ).fit(X_train, y_train).predict_proba(X_test)[:, 1]
        points = {name: {"threshold": .5} for name in POINT_NAMES}
        result = manual_sklearn_parity(
            config, X_train, y_train, X_test, np.arange(3), np.arange(3).astype(str),
            points, sklearn_scores,
        )
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["probability_differences_gt_1e_10"], 0)
        self.assertTrue(all(item["agreement_count"] == 3 for item in result["operating_points"].values()))


if __name__ == "__main__":
    unittest.main()
