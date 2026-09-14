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
        self._train_squared_norms = np.sum(self.X_ * self.X_, axis=1)
        self.y_ = y.astype(np.int64, copy=True)
        self.classes_ = np.array([0, 1])
        return self

    def _squared_distances(self, queries, workspace=None):
        with np.errstate(over="raise", invalid="raise"):
            query_squared_norms = np.sum(queries * queries, axis=1)
            if workspace is None:
                squared_distances = queries @ self.X_.T
            else:
                squared_distances = workspace[:len(queries)]
                np.matmul(queries, self.X_.T, out=squared_distances)

            squared_distances *= -2.0
            squared_distances += query_squared_norms[:, None]
            squared_distances += self._train_squared_norms[None, :]

        np.maximum(squared_distances, 0.0, out=squared_distances)
        return squared_distances

    def _direct_squared_distances(self, query):
        squared_distances = np.zeros(len(self.X_), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            for feature in range(self.X_.shape[1]):
                difference = query[feature] - self.X_[:, feature]
                squared_distances += difference * difference

        np.maximum(squared_distances, 0.0, out=squared_distances)
        return squared_distances

    def _select_exact_neighbours(self, exact_distances):
        partial_indices = np.argpartition(
            exact_distances,
            kth=self.n_neighbors - 1,
        )[:self.n_neighbors]
        threshold = exact_distances[partial_indices].max()

        closer_indices = np.flatnonzero(exact_distances < threshold)
        equal_indices = np.flatnonzero(exact_distances == threshold)
        remaining = self.n_neighbors - len(closer_indices)
        selected_indices = np.concatenate(
            (closer_indices, equal_indices[:remaining])
        )

        selected_distances = exact_distances[selected_indices]
        selected_order = np.lexsort((selected_indices, selected_distances))
        selected_indices = selected_indices[selected_order]
        selected_distances = selected_distances[selected_order]
        return selected_distances, selected_indices

    def _select_neighbours(self, squared_distances, queries):
        training_rows = len(self.X_)
        query_rows = len(queries)

        if self.n_neighbors == training_rows:
            nearest_indices = np.empty(
                (query_rows, training_rows),
                dtype=np.int64,
            )
            nearest_squared_distances = np.empty_like(
                nearest_indices,
                dtype=np.float64,
            )
            for row_number, query in enumerate(queries):
                exact_distances = self._direct_squared_distances(query)
                selected_distances, exact_indices = self._select_exact_neighbours(
                    exact_distances
                )
                nearest_indices[row_number] = exact_indices
                nearest_squared_distances[row_number] = selected_distances

            return nearest_squared_distances, nearest_indices

        nearest_indices = np.empty(
            (query_rows, self.n_neighbors),
            dtype=np.int64,
        )
        nearest_squared_distances = np.empty_like(
            nearest_indices,
            dtype=np.float64,
        )
        query_norms = np.sum(queries * queries, axis=1)
        float64_epsilon = np.finfo(np.float64).eps

        for row_number, row_distances in enumerate(squared_distances):
            candidate_indices = np.argpartition(
                row_distances,
                kth=self.n_neighbors,
            )[:self.n_neighbors + 1]
            candidate_distances = row_distances[candidate_indices]
            candidate_order = np.lexsort(
                (candidate_indices, candidate_distances)
            )
            candidate_indices = candidate_indices[candidate_order]
            candidate_distances = candidate_distances[candidate_order]

            kth_index = candidate_indices[self.n_neighbors - 1]
            next_index = candidate_indices[self.n_neighbors]
            boundary_gap = (
                candidate_distances[self.n_neighbors]
                - candidate_distances[self.n_neighbors - 1]
            )
            boundary_training_norm = max(
                self._train_squared_norms[kth_index],
                self._train_squared_norms[next_index],
            )
            boundary_scale = max(
                1.0,
                query_norms[row_number] + boundary_training_norm,
            )
            boundary_tolerance = 64 * float64_epsilon * boundary_scale

            if boundary_gap <= boundary_tolerance:
                exact_distances = self._direct_squared_distances(
                    queries[row_number]
                )
                selected_distances, selected_indices = (
                    self._select_exact_neighbours(exact_distances)
                )
            else:
                selected_indices = candidate_indices[:self.n_neighbors]
                selected_distances = candidate_distances[:self.n_neighbors]

            nearest_indices[row_number] = selected_indices
            nearest_squared_distances[row_number] = selected_distances

        return nearest_squared_distances, nearest_indices

    def _vote_probabilities(self, neighbour_indices):
        probability_one = self.y_[neighbour_indices].sum(axis=1) / self.n_neighbors
        return np.column_stack((1.0 - probability_one, probability_one))

    def _find_neighbours(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("fit must be called before prediction or neighbor queries")
        queries = self._validate_X(X)
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Query feature count differs from training feature count")

        indices = np.empty((len(queries), self.n_neighbors), dtype=np.int64)
        nearest_squared_distances = np.empty_like(indices, dtype=np.float64)
        workspace_rows = min(self.batch_size, len(queries))
        distance_workspace = np.empty(
            (workspace_rows, len(self.X_)),
            dtype=np.float64,
        )

        for start in range(0, len(queries), self.batch_size):
            batch = queries[start:start + self.batch_size]
            squared_distances = self._squared_distances(batch, distance_workspace)
            batch_squared_distances, batch_indices = self._select_neighbours(
                squared_distances,
                batch,
            )
            indices[start:start + len(batch)] = batch_indices
            nearest_squared_distances[start:start + len(batch)] = (
                batch_squared_distances
            )

        return nearest_squared_distances, indices

    def kneighbors(self, X):
        nearest_squared_distances, indices = self._find_neighbours(X)
        distances = np.sqrt(nearest_squared_distances)
        return distances, indices

    def predict_proba(self, X):
        _, neighbour_indices = self._find_neighbours(X)
        return self._vote_probabilities(neighbour_indices)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predicted_class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes_[predicted_class_indices]
        return predictions
