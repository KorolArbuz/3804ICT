import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.model_quality_v2.config import V2Configuration
from src.model_quality_v2.custom_v2_knn import CustomV2KNNClassifier
from src.model_quality_v2.features import (
    MONEY_COLUMNS, NOMINAL_COLUMNS, OTHER_NUMERIC_COLUMNS, PAY_STATUS_COLUMNS,
    safe_divide, signed_log,
)
from src.model_quality_v2.nested_cv import (
    deterministic_splits, inner_oof_selection, stage_acceptance,
)
from src.model_quality_v2.search import freeze_final_configuration
from src.model_quality_v2.thresholding import select_f1_threshold
from src.model_quality_v2.transforms import StructuredPayTransformer, V2Preprocessor


def sample_frame(rows=24):
    index = np.arange(rows, dtype=float)
    values = {}
    for column in MONEY_COLUMNS:
        values[column] = 1000 + 17 * index
    values["BILL_AMT1"] = -100 + 31 * index
    for offset, column in enumerate(PAY_STATUS_COLUMNS):
        values[column] = ((index.astype(int) + offset) % 6 - 2).astype(float)
    values["AGE"] = 21 + index % 30
    values["SEX"] = 1 + index % 2
    values["EDUCATION"] = 1 + index % 3
    values["MARRIAGE"] = 1 + index % 2
    return pd.DataFrame(values)


class FeatureTests(unittest.TestCase):
    def test_signed_log(self):
        actual = signed_log([-np.e + 1, 0, np.e - 1])
        np.testing.assert_allclose(actual, [-1, 0, 1])

    def test_safe_division_has_no_epsilon_fallback(self):
        actual = safe_divide([4, 4, 4, 4], [2, 0, -1, np.nan])
        self.assertEqual(actual[0], 2)
        self.assertTrue(np.isnan(actual[1:]).all())

    def test_yeo_johnson_is_fitted_on_supplied_fold(self):
        frame = sample_frame()
        first = V2Preprocessor(V2Configuration(money_transform="yeo_johnson")).fit(frame.iloc[:12])
        second = V2Preprocessor(V2Configuration(money_transform="yeo_johnson")).fit(frame.iloc[12:])
        a = first.transformer_.named_transformers_["money"].named_steps["scaler"].lambdas_
        b = second.transformer_.named_transformers_["money"].named_steps["scaler"].lambdas_
        self.assertFalse(np.array_equal(a, b))

    def test_unseen_pay_category_is_safe(self):
        train = np.array([[-2, 0], [0, 1], [-1, 2]], dtype=float)
        transform = StructuredPayTransformer(columns=("PAY_A", "PAY_B")).fit(train)
        result = transform.transform(np.array([[-99, -88]], dtype=float))
        category_absent = transform.transform(np.array([[3, 3]], dtype=float))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[0, 2:], category_absent[0, 2:])

    def test_group_weight_uses_square_root(self):
        frame = sample_frame()
        one = V2Preprocessor(V2Configuration(feature_blocks=("B",), pay_group_weight=1.0))
        two = V2Preprocessor(V2Configuration(feature_blocks=("B",), pay_group_weight=2.0))
        x1, x2 = one.fit_transform(frame), two.fit_transform(frame)
        np.testing.assert_allclose(x2[:, two.pay_group_indices_],
                                   x1[:, one.pay_group_indices_] * np.sqrt(2))
        other = np.setdiff1d(np.arange(x1.shape[1]), one.pay_group_indices_)
        np.testing.assert_allclose(x2[:, other], x1[:, other])

    def test_feature_names_are_unique(self):
        preprocessor = V2Preprocessor(V2Configuration(
            pay_representation="structured", feature_blocks=("A", "B"),
        )).fit(sample_frame())
        names = preprocessor.get_feature_names_out()
        self.assertEqual(len(names), len(set(names)))


class NestedProtocolTests(unittest.TestCase):
    def test_splits_are_deterministic_separated_and_complete(self):
        y = np.tile([0, 1], 15)
        first = deterministic_splits(y, 5)
        second = deterministic_splits(y, 5)
        seen = []
        for (fit1, valid1), (fit2, valid2) in zip(first, second):
            np.testing.assert_array_equal(fit1, fit2)
            np.testing.assert_array_equal(valid1, valid2)
            self.assertFalse(set(fit1).intersection(valid1))
            seen.extend(valid1)
        self.assertEqual(sorted(seen), list(range(len(y))))

    def test_outer_labels_are_not_an_inner_selector_argument(self):
        parameters = inspect.signature(inner_oof_selection).parameters
        self.assertNotIn("y_validation", parameters)
        self.assertNotIn("outer_labels", parameters)

    def test_inner_threshold_matches_only_returned_oof_scores(self):
        X = sample_frame(24)
        y = np.tile([0, 1], 12)
        config = V2Configuration(k=3)
        _, threshold, _, scores = inner_oof_selection(X, y, [config], 1, 1)
        expected, _ = select_f1_threshold(y, scores[config.configuration_id])
        self.assertEqual(threshold, expected)

    def test_stage_acceptance_rule(self):
        baseline, candidate = [], []
        for fold in range(1, 6):
            base = {"outer_fold": fold, "average_precision": .50, "roc_auc": .70,
                    "f1": .50, "recall": .50, "precision": .50,
                    "balanced_accuracy": .60}
            baseline.append(base)
            candidate.append({**base, "average_precision": .501,
                              "roc_auc": .6991, "f1": .4981})
        accepted, conditions, report = stage_acceptance(baseline, candidate)
        self.assertTrue(accepted)  # AP wins all five; AUC/F1 remain within limits.
        self.assertEqual(report["average_precision"]["wins"], 5)
        candidate[0]["roc_auc"] = .690
        self.assertFalse(stage_acceptance(baseline, candidate)[0])

    def test_configuration_freeze_refuses_overwrite(self):
        rows = pd.DataFrame([
            {"stage": 5, "outer_fold": fold, "role": "candidate",
             "average_precision": .5, "roc_auc": .7, "f1": .5,
             "recall": .5, "precision": .5}
            for fold in range(1, 6)
        ])
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory)
            frozen = freeze_final_configuration(V2Configuration(), .3, [], rows, path)
            self.assertFalse(frozen["legacy_test_evaluated"])
            with self.assertRaises(ValueError):
                freeze_final_configuration(V2Configuration(), .4, [], rows, path)


class ManualKernelTests(unittest.TestCase):
    def test_sklearn_probability_parity_and_duplicates(self):
        X = np.array([[0, 0], [1, 0], [0, 2], [3, 4], [0, 0]], dtype=float)
        y = np.array([0, 1, 1, 0, 1])
        queries = np.array([[0, 0], [.2, .3], [2, 2]], dtype=float)
        custom = CustomV2KNNClassifier(5).fit(X, y).predict_proba(queries)
        sklearn = KNeighborsClassifier(5, algorithm="brute", metric="euclidean",
                                       weights="distance").fit(X, y).predict_proba(queries)
        np.testing.assert_allclose(custom, sklearn, atol=1e-14)

    def test_zero_distance_semantics(self):
        model = CustomV2KNNClassifier(3).fit([[0], [0], [1]], [0, 1, 1])
        self.assertEqual(model.predict_proba([[0]])[0, 1], .5)

    def test_original_index_boundary_tie(self):
        model = CustomV2KNNClassifier(2).fit([[-1], [1], [2]], [0, 1, 1])
        distances, indices = model._neighbours(np.array([0.0]))
        np.testing.assert_array_equal(indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
