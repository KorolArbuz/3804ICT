from numbers import Integral

import numpy as np

from src.common.config import DEFAULT_BATCH_SIZE


class CustomKNNClassifier:
    def __init__(self, n_neighbors: int, batch_size: int = DEFAULT_BATCH_SIZE):
        invalid_neighbour_count = (
            isinstance(n_neighbors, bool)
            or not isinstance(n_neighbors, Integral)
            or n_neighbors < 1
        )
        if invalid_neighbour_count:
            raise ValueError("n_neighbors must be an integer >= 1")

        invalid_batch_size = (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or batch_size < 1
        )
        if invalid_batch_size:
            raise ValueError("batch_size must be an integer >= 1")

        self.n_neighbors = int(n_neighbors)
        self.batch_size = int(batch_size)

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be two-dimensional with at least one feature")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values (no NaN or infinity)")
        return X

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y sample counts must match")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the training sample count")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain finite binary labels 0 or 1")
        self.X_ = np.array(X, dtype=np.float64, order="C", copy=True)
        self.y_ = y.astype(np.int64, copy=True)
        self.classes_ = np.array([0, 1])
        return self

    def _squared_distances(self, queries):
        squared = np.zeros((len(queries), len(self.X_)), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            for feature in range(self.X_.shape[1]):
                difference = queries[:, feature, None] - self.X_[None, :, feature]
                squared += difference * difference
        np.maximum(squared, 0.0, out=squared)
        return squared

    def _select_neighbours(self, squared_distances):
        nearest_indices = np.argsort(
            squared_distances,
            axis=1,
            kind="stable",
        )[:, :self.n_neighbors]
        nearest_squared_distances = np.take_along_axis(
            squared_distances,
            nearest_indices,
            axis=1,
        )
        distances = np.sqrt(nearest_squared_distances)
        return distances, nearest_indices

    def _vote_probabilities(self, neighbour_indices):
        probability_one = self.y_[neighbour_indices].sum(axis=1) / self.n_neighbors
        return np.column_stack((1.0 - probability_one, probability_one))

    def kneighbors(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("fit must be called before prediction or neighbor queries")
        queries = self._validate_X(X)
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Query feature count differs from training feature count")
        indices = np.empty((len(queries), self.n_neighbors), dtype=np.int64)
        distances = np.empty_like(indices, dtype=np.float64)

        for start in range(0, len(queries), self.batch_size):
            batch = queries[start:start + self.batch_size]
            squared_distances = self._squared_distances(batch)
            batch_distances, batch_indices = self._select_neighbours(squared_distances)
            indices[start:start + len(batch)] = batch_indices
            distances[start:start + len(batch)] = batch_distances

        return distances, indices

    def predict_proba(self, X):
        _, neighbour_indices = self.kneighbors(X)
        return self._vote_probabilities(neighbour_indices)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predicted_class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes_[predicted_class_indices]
        return predictions
