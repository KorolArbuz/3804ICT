"""Threshold operating-point tests, including fold isolation and freeze guards."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.dataset import Dataset
from src.model_quality.enhanced_custom_knn import EnhancedCustomKNNClassifier
from src.model_quality import threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_DIR = PROJECT_ROOT / "results/model_quality"


def _candidate(value, *, f1=0.6, balanced_accuracy=0.7, recall=0.5, precision=0.6):
    return {
        "threshold": value,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "precision": precision,
    }


def _selection_grid(candidates):
    if not any(candidate["threshold"] == 0.5 for candidate in candidates):
        candidates = [*candidates, _candidate(0.5, f1=0, balanced_accuracy=0, recall=0, precision=0)]
    return pd.DataFrame(candidates)


def _oof_frame(size=10):
    return pd.DataFrame({
        "training_position": np.arange(size),
        "row_id": 100 + np.arange(size),
        "original_id": [f"source-{index}" for index in range(size)],
        "fold": np.arange(size) % 5 + 1,
        "y_true": np.arange(size) % 2,
        "probability_class_1": np.linspace(0, 1, size),
    })


class ThresholdRuleTests(unittest.TestCase):
    def test_threshold_rule_includes_equal_score(self):
        scores = [0, np.nextafter(0.5, 0), 0.5, np.nextafter(0.5, 1), 1]
        assert_array_equal(threshold.predict_at_threshold(scores, 0.5), [0, 0, 1, 1, 1])
        assert_array_equal(threshold.predict_at_threshold(scores, 0), np.ones(5))
        assert_array_equal(
            threshold.predict_at_threshold(scores, np.nextafter(1, np.inf)),
            np.zeros(5),
        )

    def test_invalid_probability_vectors_and_thresholds_are_rejected(self):
        for scores in ([], [[0.1, 0.2]], [np.nan], [np.inf], [-0.1], [1.1]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    threshold.predict_at_threshold(scores, 0.5)
        for value in (np.nan, np.inf, -0.1, 1.01):
            with self.subTest(threshold=value):
                with self.assertRaises(ValueError):
                    threshold.predict_at_threshold([0.2, 0.8], value)

    def test_exact_grid_covers_every_decision_breakpoint(self):
        actual = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([0, 0.2, 0.2, 0.5, 0.9, 1])
        grid = threshold.build_threshold_grid(actual, scores)
        expected = np.unique(np.r_[scores, 0, 0.5, 1, np.nextafter(1, np.inf)])
        assert_array_equal(np.sort(grid.threshold), expected)
        for row in grid.itertuples(index=False):
            predictions = (scores >= row.threshold).astype(int)
            metrics = threshold.compute_metrics(actual, predictions)
            for name in (
                "TP", "TN", "FP", "FN", "accuracy", "precision", "recall",
                "specificity", "f1", "balanced_accuracy", "predicted_positive_rate",
            ):
                assert_allclose(getattr(row, name), metrics[name], equal_nan=True)
        self.assertNotIn("roc_auc", grid.columns)
        self.assertNotIn("average_precision", grid.columns)

    def test_half_threshold_matches_argmax_when_no_probability_tie(self):
        random = np.random.RandomState(317)
        X = random.normal(size=(50, 4))
        y = np.arange(50) % 2
        queries = random.normal(size=(11, 4))
        model = EnhancedCustomKNNClassifier(25).fit(X, y)
        scores = model.predict_proba(queries)[:, 1]
        self.assertFalse(np.any(scores == 0.5))
        assert_array_equal(threshold.predict_at_threshold(scores, 0.5), model.predict(queries))

    def test_recorded_selected_model_default_predictions_are_preserved(self):
        artifact = QUALITY_DIR / "predictions_enhanced_custom.csv"
        if not artifact.exists():
            self.skipTest("Selected-model reference predictions are unavailable")
        frame = pd.read_csv(artifact, float_precision="round_trip")
        self.assertFalse(frame.probability_class_1.eq(0.5).any())
        predictions = threshold.predict_at_threshold(frame.probability_class_1, 0.5)
        assert_array_equal(predictions, frame.y_pred)
        metrics = threshold.compute_metrics(frame.y_true, predictions)
        self.assertEqual({key: metrics[key] for key in ("TN", "FP", "FN", "TP")},
                         {"TN": 4419, "FP": 254, "FN": 875, "TP": 452})

    def test_metric_counts_and_rates(self):
        result = threshold.compute_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
        self.assertEqual({key: result[key] for key in ("TN", "FP", "FN", "TP")},
                         {"TN": 2, "FP": 1, "FN": 1, "TP": 2})
        for name in ("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy"):
            self.assertAlmostEqual(result[name], 2 / 3)
        self.assertEqual(result["predicted_positive_rate"], 0.5)

    def test_metric_edge_cases_keep_undefined_rates_nan(self):
        none_positive = threshold.compute_metrics([0, 1], [0, 0])
        self.assertTrue(np.isnan(none_positive["precision"]))
        self.assertEqual(none_positive["recall"], 0)
        self.assertEqual(none_positive["f1"], 0)
        self.assertEqual(none_positive["predicted_positive_rate"], 0)
        all_positive = threshold.compute_metrics([0, 1], [1, 1])
        self.assertEqual(all_positive["recall"], 1)
        self.assertEqual(all_positive["specificity"], 0)
        no_actual_positive = threshold.compute_metrics([0, 0], [0, 0])
        self.assertTrue(np.isnan(no_actual_positive["recall"]))
        self.assertTrue(np.isnan(no_actual_positive["f1"]))
        self.assertTrue(np.isnan(no_actual_positive["balanced_accuracy"]))
        with self.assertRaises(ValueError):
            threshold.compute_metrics([], [])
        with self.assertRaises(ValueError):
            threshold.compute_metrics([0, 1], [0])


class ThresholdSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_under_candidate_order_changes(self):
        grid = threshold.build_threshold_grid(
            [0, 1, 0, 1, 1, 0, 1, 0], [0.1, 0.7, 0.2, 0.4, 0.8, 0.6, 0.9, 0.3],
        )
        expected = threshold.select_thresholds(grid)
        for seed in range(5):
            self.assertEqual(
                threshold.select_thresholds(grid.sample(frac=1, random_state=seed)), expected,
            )
        self.assertEqual(expected["default_threshold"], 0.5)
        self.assertEqual(expected["precision_floor"], 0.55)

    def test_f1_tie_break_sequence(self):
        cases = [
            ([_candidate(0.1, f1=0.6 + 1e-14), _candidate(0.5)], 0.1),
            ([_candidate(0.1, balanced_accuracy=0.8), _candidate(0.5)], 0.1),
            ([_candidate(0.1, recall=0.8), _candidate(0.5)], 0.1),
            ([_candidate(0.125), _candidate(0.375)], 0.375),
            ([_candidate(0.25), _candidate(0.75)], 0.75),
        ]
        for candidates, expected in cases:
            with self.subTest(candidates=candidates):
                chosen = threshold.select_thresholds(_selection_grid(candidates))
                self.assertEqual(chosen["f1_optimized_threshold"], expected)

    def test_precision_floor_is_inclusive_and_excludes_nan(self):
        candidates = _selection_grid([
            _candidate(0.1, precision=0.54, recall=1),
            _candidate(0.2, precision=0.55, recall=0.9),
            _candidate(0.3, precision=0.8, recall=0.7),
            _candidate(0.9, precision=np.nan, recall=1),
        ])
        self.assertEqual(threshold.select_thresholds(candidates)["recall_oriented_threshold"], 0.2)

    def test_recall_tie_break_sequence(self):
        cases = [
            ([_candidate(0.1, recall=0.5 + 1e-14), _candidate(0.5)], 0.1),
            ([_candidate(0.1, f1=0.7), _candidate(0.5)], 0.1),
            ([_candidate(0.1, balanced_accuracy=0.8), _candidate(0.5)], 0.1),
            ([_candidate(0.25), _candidate(0.75)], 0.75),
        ]
        for candidates, expected in cases:
            with self.subTest(candidates=candidates):
                chosen = threshold.select_thresholds(_selection_grid(candidates))
                self.assertEqual(chosen["recall_oriented_threshold"], expected)

    def test_impossible_precision_floor_produces_no_recall_candidate(self):
        candidates = pd.DataFrame([
            _candidate(0.1, precision=0.3), _candidate(0.5, precision=0.5),
            _candidate(1.0, precision=np.nan),
        ])
        chosen = threshold.select_thresholds(candidates, precision_floor=0.55)
        self.assertIsNone(chosen["recall_oriented_threshold"])
        self.assertIsInstance(chosen["f1_optimized_threshold"], float)


class OOFValidationTests(unittest.TestCase):
    def test_complete_oof_positions_and_ids_are_accepted(self):
        frame = _oof_frame()
        threshold.validate_oof(frame, 10, frame.row_id.to_numpy())

    def test_missing_duplicate_or_out_of_range_positions_are_rejected(self):
        original = _oof_frame()
        invalid_frames = [original.iloc[:-1].copy()]
        for column, value in (("training_position", 0), ("training_position", 10), ("row_id", 100)):
            frame = original.copy()
            frame.loc[9, column] = value
            invalid_frames.append(frame)
        invalid_frames.append(original.drop(columns="probability_class_1"))
        for frame in invalid_frames:
            with self.subTest(columns=list(frame.columns), size=len(frame)):
                with self.assertRaises(ValueError):
                    threshold.validate_oof(frame, 10)

    def test_invalid_oof_values_and_row_mapping_are_rejected(self):
        for column, value in (
            ("probability_class_1", np.nan), ("probability_class_1", -0.1),
            ("probability_class_1", 1.1), ("fold", 0), ("fold", 6), ("y_true", 2),
        ):
            with self.subTest(column=column, value=value):
                frame = _oof_frame()
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    threshold.validate_oof(frame, 10)
        frame = _oof_frame()
        with self.assertRaises(ValueError):
            threshold.validate_oof(frame, 10, frame.row_id.to_numpy()[::-1])

    def test_oof_fits_each_preprocessor_only_on_fold_train_and_never_reads_test(self):
        train_indices = np.arange(0, 100, 2)
        markers = np.full(100, -999999, dtype=float)
        markers[train_indices] = np.arange(len(train_indices))
        labels = np.zeros(100, dtype=int)
        labels[train_indices] = np.arange(len(train_indices)) % 2
        dataset = Dataset(
            pd.DataFrame({"marker": markers}), labels, np.arange(100) + 1000,
            np.array([f"source-{index}" for index in range(100)]), {},
        )
        records = []

        class PreprocessorSpy:
            def __init__(self, frame):
                self.assert_training_only(frame)
                self.record = {"fit": None, "validation": None}
                records.append(self.record)

            @staticmethod
            def assert_training_only(frame):
                if (frame.marker < 0).any():
                    raise AssertionError("Held-out test row was accessed")

            def fit_transform(self, frame):
                self.assert_training_only(frame)
                self.record["fit"] = set(frame.marker.astype(int))
                return frame.to_numpy(dtype=float)

            def transform(self, frame):
                self.assert_training_only(frame)
                self.record["validation"] = set(frame.marker.astype(int))
                return frame.to_numpy(dtype=float)

        class NeighbourSpy:
            classes_ = np.array([0, 1])

            def __init__(self, **kwargs):
                if kwargs != {
                    "n_neighbors": 25, "metric": "euclidean", "weights": "distance",
                    "algorithm": "brute", "n_jobs": 1,
                }:
                    raise AssertionError(f"Unexpected model configuration: {kwargs}")

            def fit(self, X, y):
                if len(X) != 40 or (X < 0).any():
                    raise AssertionError("Model fit did not use fold-training rows only")
                assert_array_equal(y, X[:, 0].astype(int) % 2)
                return self

            def predict_proba(self, X):
                if len(X) != 10 or (X < 0).any():
                    raise AssertionError("Model scores did not use fold-validation rows only")
                probability = X[:, 0] / 50
                return np.column_stack((1 - probability, probability))

        with patch.object(threshold, "make_preprocessor", PreprocessorSpy), patch.object(
            threshold, "KNeighborsClassifier", NeighbourSpy,
        ):
            first = threshold.generate_oof(dataset, train_indices)
            second = threshold.generate_oof(dataset, train_indices)
        assert_frame_equal(first, second)
        threshold.validate_oof(first, 50, dataset.row_ids[train_indices])
        assert_array_equal(first.training_position, np.arange(50))
        assert_allclose(first.probability_class_1, np.arange(50) / 50)
        assert_array_equal(first.y_true, labels[train_indices])
        assert_array_equal(first.original_id, dataset.original_ids[train_indices])
        self.assertEqual(len(records), 10)
        for run_records in (records[:5], records[5:]):
            validation_rows = []
            for record in run_records:
                self.assertEqual(len(record["fit"]), 40)
                self.assertEqual(len(record["validation"]), 10)
                self.assertFalse(record["fit"] & record["validation"])
                self.assertEqual(record["fit"] | record["validation"], set(range(50)))
                validation_rows.extend(record["validation"])
            self.assertEqual(sorted(validation_rows), list(range(50)))

    def test_saved_oof_contains_exactly_24000_training_rows_and_no_test_rows(self):
        artifact = QUALITY_DIR / "threshold_study/oof_scores.csv"
        if not artifact.exists():
            self.skipTest("Full-data OOF artifact has not been generated")
        frame = pd.read_csv(artifact, float_precision="round_trip", dtype={"original_id": str})
        threshold.validate_oof(frame, 24_000)
        self.assertEqual(frame.fold.value_counts().to_dict(), {fold: 4800 for fold in range(1, 6)})
        test_rows = pd.read_csv(QUALITY_DIR / "predictions_enhanced_custom.csv")
        self.assertFalse(set(frame.row_id) & set(test_rows.row_id))
        self.assertEqual(set(frame.row_id) | set(test_rows.row_id), set(range(30_000)))


class ThresholdFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.quality_dir = Path(self.temporary.name)
        self.study_dir = self.quality_dir / "threshold_study"
        self.study_dir.mkdir()
        self.dataset_path = self.quality_dir / "dataset.csv"
        self.dataset_path.write_text("training dataset identity\n", encoding="utf-8")
        self.cv_path = self.quality_dir / "cv_quality_grid.csv"
        self.cv_path.write_text("frozen training-only CV result\n", encoding="utf-8")
        self.model_path = self.quality_dir / "selected_quality_parameters.json"
        self.model = {
            **threshold.FROZEN_MODEL, "test_evaluated": True,
            "cv_results_sha256": threshold.sha256(self.cv_path),
        }
        self.write_json(self.model_path, self.model)
        source_name = "src/custom_knn/classifier.py"
        source_hash = threshold.sha256(PROJECT_ROOT / source_name)
        self.write_json(self.quality_dir / "baseline_manifest.json", {
            "baseline_source_sha256": source_hash, "training_rows": 10,
        })
        oof = _oof_frame()
        grid = threshold.build_threshold_grid(oof.y_true, oof.probability_class_1)
        for name, frame in (("oof_scores.csv", oof), ("threshold_grid.csv", grid)):
            frame.to_csv(self.study_dir / name, index=False, float_format="%.17g")
        self.selection = {
            "frozen_model": dict(threshold.FROZEN_MODEL),
            "model_selection_sha256": threshold.sha256(self.model_path),
            "source_hashes": {source_name: source_hash},
            "artifact_hashes": {
                name: threshold.sha256(self.study_dir / name)
                for name in ("oof_scores.csv", "threshold_grid.csv")
            },
            "dataset_sha256": threshold.sha256(self.dataset_path),
            "oof_rows": 10,
            **threshold.select_thresholds(grid),
            "test_evaluated": False,
        }
        self.save_selection()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def save_selection(self):
        self.selection["freeze_sha256"] = threshold._selection_digest(self.selection)
        self.write_json(self.study_dir / "threshold_selection.json", self.selection)

    def load_selection(self, verify_rerun=False):
        return threshold.load_threshold_selection(
            self.study_dir, self.quality_dir, verify_rerun=verify_rerun,
            data_path=self.dataset_path,
        )

    def test_first_evaluation_requires_frozen_unevaluated_selection(self):
        loaded = self.load_selection()
        self.assertFalse(loaded["test_evaluated"])
        self.selection["test_evaluated"] = True
        self.save_selection()
        with self.assertRaises(ValueError):
            self.load_selection()

    def test_verify_rerun_requires_completed_selection_and_same_thresholds(self):
        with self.assertRaises(ValueError):
            self.load_selection(verify_rerun=True)
        self.selection["test_evaluated"] = True
        self.save_selection()
        self.assertTrue(self.load_selection(verify_rerun=True)["test_evaluated"])
        self.selection["f1_optimized_threshold"] = 0.123456
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "thresholds"):
            self.load_selection(verify_rerun=True)

    def test_editing_frozen_threshold_without_rehashing_is_rejected(self):
        self.selection["f1_optimized_threshold"] = 0.123456
        self.write_json(self.study_dir / "threshold_selection.json", self.selection)
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load_selection()

    def test_modifying_oof_or_grid_bytes_is_rejected(self):
        for name in ("oof_scores.csv", "threshold_grid.csv"):
            with self.subTest(name=name):
                path = self.study_dir / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                try:
                    with self.assertRaisesRegex(ValueError, "OOF artifact"):
                        self.load_selection()
                finally:
                    path.write_bytes(original)

    def test_modified_source_hash_or_dataset_is_rejected(self):
        source_name = next(iter(self.selection["source_hashes"]))
        self.selection["source_hashes"][source_name] = "0" * 64
        self.save_selection()
        with self.assertRaisesRegex(ValueError, "source"):
            self.load_selection()
        self.selection["source_hashes"][source_name] = threshold.sha256(PROJECT_ROOT / source_name)
        self.save_selection()
        self.dataset_path.write_text("different dataset\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Dataset"):
            self.load_selection()

    def test_model_configuration_cv_and_baseline_source_must_still_match(self):
        for key, value in (
            ("selected_k", 19), ("metric", "manhattan"), ("weights", "uniform"),
            ("test_evaluated", False), ("cv_results_sha256", "0" * 64),
        ):
            with self.subTest(key=key):
                self.write_json(self.model_path, {**self.model, key: value})
                with self.assertRaises(ValueError):
                    threshold.verify_model(self.quality_dir)
        self.write_json(self.model_path, self.model)
        self.write_json(self.quality_dir / "baseline_manifest.json", {
            "baseline_source_sha256": "0" * 64,
        })
        with self.assertRaisesRegex(ValueError, "V5.1"):
            threshold.verify_model(self.quality_dir)

    def test_oof_overwrite_guard_runs_before_dataset_loading(self):
        with patch.object(threshold, "load_dataset") as loader:
            with self.assertRaises(ValueError):
                threshold.run_oof(
                    data=self.dataset_path, quality_dir=self.quality_dir, output_dir=self.study_dir,
                )
            loader.assert_not_called()

    def test_oof_does_not_overwrite_partial_nonempty_output_directory(self):
        output_dir = self.quality_dir / "partial_study"
        output_dir.mkdir()
        partial = output_dir / "oof_scores.csv"
        partial.write_bytes(b"partial output must survive\n")
        with patch.object(threshold, "load_dataset") as loader:
            with self.assertRaises(ValueError):
                threshold.run_oof(
                    data=self.dataset_path, quality_dir=self.quality_dir, output_dir=output_dir,
                )
            loader.assert_not_called()
        self.assertEqual(partial.read_bytes(), b"partial output must survive\n")

    def test_test_evaluation_guard_runs_before_dataset_loading(self):
        self.selection["test_evaluated"] = True
        self.save_selection()
        with patch.object(threshold, "load_dataset") as loader:
            with self.assertRaises(ValueError):
                threshold.evaluate_thresholds(
                    data=self.dataset_path, quality_dir=self.quality_dir, output_dir=self.study_dir,
                )
            loader.assert_not_called()

    def test_missing_selection_is_rejected_before_dataset_loading(self):
        (self.study_dir / "threshold_selection.json").unlink()
        with patch.object(threshold, "load_dataset") as loader:
            with self.assertRaises((ValueError, FileNotFoundError)):
                threshold.evaluate_thresholds(
                    data=self.dataset_path, quality_dir=self.quality_dir, output_dir=self.study_dir,
                )
            loader.assert_not_called()


class ThresholdBootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_matches_independent_rowwise_oracle_and_repeats(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.3, 0.6, 0.2, 0.4, 0.8])
        selection = {
            "default_threshold": 0.5,
            "f1_optimized_threshold": 0.35,
            "recall_oriented_threshold": 0.2,
        }
        resamples, seed = 413, 173
        first = threshold.paired_bootstrap(labels, scores, selection, resamples=resamples, seed=seed)
        second = threshold.paired_bootstrap(labels, scores, selection, resamples=resamples, seed=seed)
        assert_frame_equal(first, second)

        def rowwise_metrics(actual, predicted):
            counts = {
                (actual_class, predicted_class): sum(
                    a == actual_class and p == predicted_class for a, p in zip(actual, predicted)
                )
                for actual_class in (0, 1) for predicted_class in (0, 1)
            }
            tn, fp, fn, tp = (counts[(0, 0)], counts[(0, 1)], counts[(1, 0)], counts[(1, 1)])

            def ratio(numerator, denominator):
                return numerator / denominator if denominator else np.nan

            return {
                "TP": tp, "FP": fp, "FN": fn,
                "f1": ratio(2 * tp, 2 * tp + fp + fn),
                "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
                "balanced_accuracy": (ratio(tp, tp + fn) + ratio(tn, tn + fp)) / 2,
            }

        predictions = {name: scores >= value for name, value in selection.items()}
        candidates = ["f1_optimized_threshold", "recall_oriented_threshold"]
        random = np.random.default_rng(seed)
        samples = []
        for _ in range(resamples):
            positions = random.integers(0, len(labels), size=len(labels))
            samples.append({
                name: rowwise_metrics(labels[positions], predicted[positions])
                for name, predicted in predictions.items()
            })
        observed = {name: rowwise_metrics(labels, predicted) for name, predicted in predictions.items()}
        for name in candidates:
            for metric in observed[name]:
                with self.subTest(candidate=name, metric=metric):
                    values = np.array([
                        sample[name][metric] - sample["default_threshold"][metric]
                        for sample in samples
                    ])
                    finite = values[np.isfinite(values)]
                    bounds = np.percentile(finite, [2.5, 97.5])
                    row = first[(first.threshold_name == name) & (first.metric == metric)].iloc[0]
                    self.assertAlmostEqual(
                        row.observed_delta,
                        observed[name][metric] - observed["default_threshold"][metric],
                    )
                    assert_allclose(
                        [row.bootstrap_95_percentile_lower, row.bootstrap_95_percentile_upper],
                        bounds,
                    )
                    self.assertEqual(row.valid_resamples, len(finite))
                    self.assertEqual(row.resamples, resamples)
                    self.assertEqual(row.seed, seed)


if __name__ == "__main__":
    unittest.main()
