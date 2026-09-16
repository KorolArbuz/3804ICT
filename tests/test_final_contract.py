import unittest
import numpy as np

from src.common.experiment import array_hash, load_configs, sha256, BASELINE_SHA256
from src.common.config import PROJECT_ROOT
from src.evaluation.final import prediction_frame, validate_predictions
from src.final_knn.classifier import CustomV2KNNClassifier


class FinalContractTests(unittest.TestCase):
    def test_baseline_bytes(self):
        self.assertEqual(
            sha256(PROJECT_ROOT / "src/custom_knn/classifier.py"), BASELINE_SHA256
        )

    def test_exact_distance_zero_and_boundary(self):
        X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]])
        y = np.array([0, 1, 1, 0, 1])
        model = CustomV2KNNClassifier(3).fit(X, y)
        self.assertEqual(model.predict_proba([[0, 0]])[0, 1], 0.5)
        distance, index = model._neighbours(np.array([0.0, 0.0]))
        np.testing.assert_array_equal(index, [0, 1, 2])
        # Independent full-sort oracle (distance, original training position).
        rng = np.random.default_rng(721)
        X = rng.normal(size=(57, 9))
        y = rng.integers(0, 2, 57)
        queries = rng.normal(size=(11, 9))
        model = CustomV2KNNClassifier(17).fit(X, y)
        actual = model.predict_proba(queries)[:, 1]
        expected = []
        for query in queries:
            distances = np.sqrt(np.sum((X - query) ** 2, axis=1))
            indices = sorted(range(len(X)), key=lambda i: (distances[i], i))[:17]
            weights = 1 / distances[indices]
            expected.append(sum(weights * y[indices]) / sum(weights))
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)

    def test_invalid_inputs(self):
        for k in (0, True, 1.5):
            with self.assertRaises(ValueError):
                CustomV2KNNClassifier(k)
        with self.assertRaises(ValueError):
            CustomV2KNNClassifier(2).fit([[0]], [1])
        with self.assertRaises(ValueError):
            CustomV2KNNClassifier(1).fit([[np.inf]], [1])
        model = CustomV2KNNClassifier(1).fit([[0]], [1])
        with self.assertRaises(ValueError):
            model.predict([[1, 2]])
        self.assertEqual(model.predict_proba(np.empty((0, 1))).shape, (0, 2))

    def test_k_sizes_copy_repeated_predictions_and_test_label_independence(self):
        rng = np.random.default_rng(201)
        training = rng.normal(size=(121, 3))
        labels = rng.integers(0, 2, 121)
        queries = rng.normal(size=(7, 3))
        for k in (1, 2, 19, 101, 121):
            with self.subTest(k=k):
                X = training.copy()
                y = labels.copy()
                model = CustomV2KNNClassifier(k).fit(X, y)
                before = model.predict_proba(queries)
                X[:] = 999
                y[:] = 0
                np.testing.assert_array_equal(before, model.predict_proba(queries))
                np.testing.assert_allclose(before.sum(axis=1), 1, atol=0, rtol=0)
                self.assertTrue(np.isfinite(before).all())
        # Test labels are absent from the scoring API; changing an evaluator-side array has no effect.
        test_labels = np.zeros(len(queries))
        model = CustomV2KNNClassifier(19).fit(training, labels)
        first = model.predict_proba(queries)
        test_labels[:] = 1
        np.testing.assert_array_equal(first, model.predict_proba(queries))
        duplicate = CustomV2KNNClassifier(2).fit([[0], [0], [0], [0]], [0, 1, 1, 1])
        self.assertEqual(duplicate.predict_proba([[0]])[0, 1], 0.5)
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            CustomV2KNNClassifier(1).fit([[1e308]], [0]).predict([[-1e308]])

    def test_threshold_exact_and_next_representable(self):
        config = load_configs()["final"]
        threshold = config["model"]["threshold"]
        scores = np.array(
            [
                np.nextafter(threshold, -np.inf),
                threshold,
                np.nextafter(threshold, np.inf),
            ]
        )
        data = {
            "y_test": np.array([0, 1, 0], dtype=np.int64),
            "row_ids": np.array([1, 4, 9], dtype=np.int64),
            "original_ids": np.array(["a", "b", "c"]),
        }
        manifest = {
            "split": {
                "test_row_ids_hash": array_hash(data["row_ids"]),
                "y_test_hash": array_hash(data["y_test"]),
                "original_test_ids_hash": array_hash(data["original_ids"]),
            }
        }
        frame = prediction_frame(
            "exact", "final_cpp", config, data, scores, np.array([0, 1, 1])
        )
        validate_predictions(frame, "exact", "final_cpp", config, manifest)
        frame.loc[1, "y_pred"] = 0
        with self.assertRaises(ValueError):
            validate_predictions(frame, "exact", "final_cpp", config, manifest)

    def test_predictions_reject_stale_misaligned_invalid(self):
        config = load_configs()["final"]
        data = {
            "y_test": np.array([0, 1], dtype=np.int64),
            "row_ids": np.array([10, 20], dtype=np.int64),
            "original_ids": np.array(["11", "21"]),
        }
        manifest = {
            "split": {
                "test_row_ids_hash": array_hash(data["row_ids"]),
                "y_test_hash": array_hash(data["y_test"]),
                "original_test_ids_hash": array_hash(data["original_ids"]),
            }
        }
        frame = prediction_frame(
            "run", "final_python", config, data, np.array([0.2, 0.6]), np.array([0, 1])
        )
        validate_predictions(frame, "run", "final_python", config, manifest)
        for column, value in [
            ("run_id", "old"),
            ("config_hash", "stale"),
            ("row_id", 999),
            ("original_id", "bad"),
            ("score_class_1", np.nan),
            ("y_pred", 1),
            ("threshold", 0.5),
            ("test_position", 10),
        ]:
            bad = frame.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_predictions(bad, "run", "final_python", config, manifest)


if __name__ == "__main__":
    unittest.main()
