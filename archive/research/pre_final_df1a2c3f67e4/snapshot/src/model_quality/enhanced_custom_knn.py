"""Manual exact KNN for the frozen Euclidean/distance-weighted CV winner."""

from numbers import Integral

import numpy as np

from src.common.config import DEFAULT_BATCH_SIZE


class EnhancedCustomKNNClassifier:
    """Exact brute-force Euclidean KNN with inverse-distance voting.

    This class intentionally implements only the model family selected by the
    training-only search: Euclidean distance and distance weighting. ``k`` is
    configurable so boundary behavior can be unit-tested independently of the
    final selected value (25).
    """

    metric = "euclidean"
    weights = "distance"
    classes_ = np.array([0, 1], dtype=np.int64)

    def __init__(self, n_neighbors, batch_size=DEFAULT_BATCH_SIZE):
        if (
            isinstance(n_neighbors, bool)
            or not isinstance(n_neighbors, Integral)
            or n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be an integer >= 1")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be an integer >= 1")
        self.n_neighbors = int(n_neighbors)
        self.batch_size = int(batch_size)

    @staticmethod
    def _validate_X(X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("X must be two-dimensional with at least one feature")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values (no NaN or infinity)")
        return values

    def fit(self, X, y):
        values = self._validate_X(X)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(values) != len(labels):
            raise ValueError("X and y sample counts must match")
        if self.n_neighbors > len(values):
            raise ValueError("n_neighbors cannot exceed the training sample count")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("y must contain finite binary labels 0 or 1")

        self.X_ = np.array(values, dtype=np.float64, order="C", copy=True)
        self.y_ = labels.astype(np.int64, copy=True)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def _exact_squared_distances(self, query, indices=None):
        training = self.X_ if indices is None else self.X_[indices]
        with np.errstate(over="raise", invalid="raise"):
            differences = training - query
            distances = np.einsum("ij,ij->i", differences, differences)
        return distances

    def _select_from_exact(self, squared_distances):
        """Choose exact top-k, resolving boundary ties by training index."""
        if self.n_neighbors == len(squared_distances):
            candidates = np.arange(len(squared_distances), dtype=np.int64)
        else:
            partial = np.argpartition(
                squared_distances, self.n_neighbors - 1
            )[: self.n_neighbors]
            boundary = squared_distances[partial].max()
            closer = np.flatnonzero(squared_distances < boundary)
            equal = np.flatnonzero(squared_distances == boundary)
            candidates = np.concatenate(
                (closer, equal[: self.n_neighbors - len(closer)])
            )
        order = np.lexsort((candidates, squared_distances[candidates]))
        return candidates[order]

    def kneighbors(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("fit must be called before prediction")
        queries = self._validate_X(X)
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Query feature count differs from training feature count")

        all_indices = np.empty((len(queries), self.n_neighbors), dtype=np.int64)
        all_distances = np.empty((len(queries), self.n_neighbors), dtype=np.float64)
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            for offset, query in enumerate(batch):
                # Compute every distance directly from coordinate differences.
                # This is intentionally clarity-first exact brute force; no
                # approximate index or matrix-product prefilter is used.
                exact_squared_distances = self._exact_squared_distances(query)
                indices = self._select_from_exact(exact_squared_distances)
                row = start + offset
                all_indices[row] = indices
                all_distances[row] = np.sqrt(exact_squared_distances[indices])
        return all_distances, all_indices

    def predict_proba(self, X):
        distances, indices = self.kneighbors(X)
        neighbour_labels = self.y_[indices]
        probability_one = np.empty(len(indices), dtype=np.float64)
        for row, (row_distances, row_labels) in enumerate(
            zip(distances, neighbour_labels)
        ):
            zero = row_distances == 0.0
            if zero.any():
                # Any exact match suppresses all nonzero-distance neighbours;
                # zero matches then share equal effective weight.
                probability_one[row] = row_labels[zero].mean()
            else:
                inverse = 1.0 / row_distances
                probability_one[row] = np.dot(inverse, row_labels) / inverse.sum()
        return np.column_stack((1.0 - probability_one, probability_one))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
