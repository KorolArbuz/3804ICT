import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import KNeighborsClassifier

from src.model_quality.enhanced_custom_knn import EnhancedCustomKNNClassifier
from src.model_quality.selection import aggregate_cv, select_configuration


class EnhancedCustomKNNTests(unittest.TestCase):
    def compare_with_sklearn(self, X, y, queries, k):
        custom = EnhancedCustomKNNClassifier(k).fit(X, y)
        sklearn = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="distance",
            algorithm="brute",
            n_jobs=1,
        ).fit(X, y)
        assert_array_equal(custom.predict(queries), sklearn.predict(queries))
        assert_allclose(
            custom.predict_proba(queries),
            sklearn.predict_proba(queries),
            rtol=2e-14,
            atol=2e-14,
        )

    def test_k_one(self):
        self.compare_with_sklearn(
            np.array([[0.0], [2.0], [5.0]]),
            np.array([0, 1, 0]),
            np.array([[1.8], [4.0]]),
            1,
        )

    def test_multiple_zero_distance_rows_with_mixed_labels(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 1, 0])
        query = np.array([[1.0, 2.0]])
        self.compare_with_sklearn(X, y, query, 4)
        probabilities = EnhancedCustomKNNClassifier(4).fit(X, y).predict_proba(query)
        assert_allclose(probabilities, [[0.5, 0.5]])

    def test_single_zero_distance_suppresses_other_neighbours(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([1, 0, 0])
        probabilities = EnhancedCustomKNNClassifier(3).fit(X, y).predict_proba([[0.0]])
        assert_allclose(probabilities, [[0.0, 1.0]])

    def test_equal_distance_boundary_uses_original_training_index(self):
        X = np.array([[-1.0], [1.0], [-2.0], [2.0]])
        y = np.array([1, 0, 0, 1])
        classifier = EnhancedCustomKNNClassifier(1).fit(X, y)
        distances, indices = classifier.kneighbors([[0.0]])
        assert_array_equal(indices, [[0]])
        assert_allclose(distances, [[1.0]])
        assert_array_equal(classifier.predict([[0.0]]), [1])

    def test_repeatability_and_random_sklearn_parity(self):
        random = np.random.RandomState(42)
        X = random.normal(size=(80, 5))
        y = random.randint(0, 2, size=80)
        queries = random.normal(size=(17, 5))
        classifier = EnhancedCustomKNNClassifier(25, batch_size=4).fit(X, y)
        first = classifier.predict_proba(queries)
        second = classifier.predict_proba(queries)
        assert_array_equal(first, second)
        self.compare_with_sklearn(X, y, queries, 25)

    def test_invalid_k_feature_count_nan_and_infinity(self):
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                EnhancedCustomKNNClassifier(invalid)
        classifier = EnhancedCustomKNNClassifier(1).fit([[0.0, 1.0]], [0])
        with self.assertRaises(ValueError):
            classifier.predict([[0.0]])
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                classifier.predict([[invalid, 0.0]])


class SelectionTests(unittest.TestCase):
    def test_selection_is_f1_first_and_deterministic(self):
        rows = []
        metric_names = (
            "accuracy", "precision", "recall", "specificity", "f1",
            "balanced_accuracy", "roc_auc", "average_precision",
        )
        for metric in ("euclidean", "manhattan"):
            for weights in ("uniform", "distance"):
                for k in range(1, 32, 2):
                    for fold in range(1, 6):
                        rows.append({
                            "fold": fold,
                            "k": k,
                            "metric": metric,
                            "weights": weights,
                            **{name: 0.5 for name in metric_names},
                            "TN": 1,
                            "FP": 1,
                            "FN": 1,
                            "TP": 1,
                        })
        import pandas as pd

        winner = select_configuration(aggregate_cv(pd.DataFrame(rows)))
        self.assertEqual(
            (winner.k, winner.metric, winner.weights),
            (1, "euclidean", "uniform"),
        )


if __name__ == "__main__":
    unittest.main()
