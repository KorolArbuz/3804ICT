"""Clarity-first exact KNN kernel for the frozen V2 transformed matrix."""

from numbers import Integral

import numpy as np


class CustomV2KNNClassifier:
    metric = "euclidean"
    weights = "distance"

    def __init__(self, n_neighbors):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral) or n_neighbors < 1:
            raise ValueError("n_neighbors must be an integer >= 1")
        self.n_neighbors = int(n_neighbors)

    @staticmethod
    def _X(X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("X must be a finite two-dimensional matrix with features")
        return values

    def fit(self, X, y):
        values = self._X(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values) or not np.isin(labels, [0, 1]).all():
            raise ValueError("y must be aligned binary labels")
        if self.n_neighbors > len(values):
            raise ValueError("n_neighbors exceeds training rows")
        self.X_ = np.array(values, order="C", copy=True)
        self.y_ = labels.astype(np.int64, copy=True)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def _neighbours(self, query):
        differences = self.X_ - query
        squared = np.einsum("ij,ij->i", differences, differences)
        if self.n_neighbors == len(squared):
            candidates = np.arange(len(squared))
        else:
            partial = np.argpartition(squared, self.n_neighbors - 1)[:self.n_neighbors]
            boundary = squared[partial].max()
            closer = np.flatnonzero(squared < boundary)
            equal = np.flatnonzero(squared == boundary)
            candidates = np.r_[closer, equal[:self.n_neighbors - len(closer)]]
        order = np.lexsort((candidates, squared[candidates]))
        indices = candidates[order]
        return np.sqrt(squared[indices]), indices

    def predict_proba(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("fit must be called before prediction")
        queries = self._X(X)
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Query feature count differs from training")
        probability_one = np.empty(len(queries))
        for row, query in enumerate(queries):
            distances, indices = self._neighbours(query)
            labels = self.y_[indices]
            zero = distances == 0
            if zero.any():
                probability_one[row] = labels[zero].mean()
            else:
                inverse = 1.0 / distances
                probability_one[row] = np.dot(inverse, labels) / inverse.sum()
        return np.column_stack((1 - probability_one, probability_one))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
