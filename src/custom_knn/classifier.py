from numbers import Integral

import numpy as np

from src.common.config import DEFAULT_BATCH_SIZE


class CustomKNNClassifier:
    _PREFILTER_PILOT_SIZE = 1024
    _PREFILTER_DENSITY_LIMIT = 0.5

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
        self._augmented_training = np.empty(
            (len(self.X_), self.X_.shape[1] + 1),
            dtype=np.float64,
        )
        self._augmented_training[:, :-1] = -2.0 * self.X_
        self._augmented_training[:, -1] = self._train_squared_norms
        self._X_by_feature = np.ascontiguousarray(self.X_.T)
        self.y_ = y.astype(np.int64, copy=True)
        self.classes_ = np.array([0, 1])

        training_rows = len(self.X_)
        pilot_size = min(self._PREFILTER_PILOT_SIZE, training_rows)
        can_prefilter = (
            self.n_neighbors < training_rows
            and pilot_size >= self.n_neighbors + 1
            and pilot_size * 2 <= training_rows
        )
        if can_prefilter:
            self._pilot_indices = (
                np.arange(pilot_size, dtype=np.int64)
                * training_rows
                // pilot_size
            )
        else:
            self._pilot_indices = None
        return self

    def _squared_distances(self, queries, query_squared_norms, workspace=None):
        with np.errstate(over="raise", invalid="raise"):
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

    def _ranking_scores(self, queries, workspace):
        # Appending 1 to each query folds the training norm into the matrix
        # product. The result is ||x||^2 - 2(q dot x), which has the same
        # ranking and gap units as squared distance after omitting ||q||^2.
        scores = workspace[:len(queries)]
        augmented_queries = np.empty(
            (len(queries), queries.shape[1] + 1),
            dtype=np.float64,
        )
        augmented_queries[:, :-1] = queries
        augmented_queries[:, -1] = 1.0
        with np.errstate(over="raise", invalid="raise"):
            np.matmul(
                augmented_queries,
                self._augmented_training.T,
                out=scores,
            )
        return scores

    def _direct_squared_distances(self, query):
        squared_distances = np.zeros(len(self.X_), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            for feature in range(self.X_.shape[1]):
                difference = query[feature] - self._X_by_feature[feature]
                squared_distances += difference * difference

        np.maximum(squared_distances, 0.0, out=squared_distances)
        return squared_distances

    def _select_exact_neighbours(self, exact_distances, return_distances):
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

        if not return_distances:
            return None, selected_indices

        selected_distances = exact_distances[selected_indices]
        selected_order = np.lexsort((selected_indices, selected_distances))
        selected_indices = selected_indices[selected_order]
        selected_distances = selected_distances[selected_order]
        return selected_distances, selected_indices

    def _select_neighbours(
        self,
        ranking_values,
        queries,
        query_squared_norms,
        return_distances,
    ):
        training_rows = len(self.X_)
        query_rows = len(queries)

        if self.n_neighbors == training_rows:
            if not return_distances:
                all_indices = np.arange(training_rows, dtype=np.int64)
                nearest_indices = np.broadcast_to(
                    all_indices,
                    (query_rows, training_rows),
                ).copy()
                return None, nearest_indices

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
                    exact_distances,
                    return_distances=True,
                )
                nearest_indices[row_number] = exact_indices
                nearest_squared_distances[row_number] = selected_distances

            return nearest_squared_distances, nearest_indices

        nearest_indices = np.empty(
            (query_rows, self.n_neighbors),
            dtype=np.int64,
        )
        nearest_squared_distances = None
        if return_distances:
            nearest_squared_distances = np.empty_like(
                nearest_indices,
                dtype=np.float64,
            )

        float64_epsilon = np.finfo(np.float64).eps

        for row_number, row_values in enumerate(ranking_values):
            selection_values = row_values
            source_indices = None

            if not return_distances and self._pilot_indices is not None:
                pilot_values = row_values[self._pilot_indices]
                threshold = np.partition(
                    pilot_values,
                    kth=self.n_neighbors,
                )[self.n_neighbors]

                # The pilot contains k+1 distinct training rows at or below
                # this threshold. Scanning with <= therefore retains the full
                # global top-k boundary, including every threshold tie.
                filtered_indices = np.flatnonzero(row_values <= threshold)
                if (
                    len(filtered_indices) >= self.n_neighbors + 1
                    and len(filtered_indices)
                    <= self._PREFILTER_DENSITY_LIMIT * training_rows
                ):
                    source_indices = filtered_indices
                    selection_values = row_values[filtered_indices]

            local_candidates = np.argpartition(
                selection_values,
                kth=self.n_neighbors,
            )[:self.n_neighbors + 1]
            candidate_values = selection_values[local_candidates]
            if source_indices is None:
                candidate_indices = local_candidates
            else:
                candidate_indices = source_indices[local_candidates]
            selected_indices = candidate_indices[:self.n_neighbors]
            selected_candidate_values = candidate_values[:self.n_neighbors]
            kth_position = np.argmax(selected_candidate_values)
            kth_index = selected_indices[kth_position]
            next_index = candidate_indices[self.n_neighbors]
            boundary_gap = (
                candidate_values[self.n_neighbors]
                - selected_candidate_values[kth_position]
            )
            boundary_training_norm = max(
                self._train_squared_norms[kth_index],
                self._train_squared_norms[next_index],
            )
            boundary_scale = max(
                1.0,
                query_squared_norms[row_number] + boundary_training_norm,
            )
            boundary_tolerance = 64 * float64_epsilon * boundary_scale

            if boundary_gap <= boundary_tolerance:
                exact_distances = self._direct_squared_distances(
                    queries[row_number]
                )
                selected_distances, selected_indices = (
                    self._select_exact_neighbours(
                        exact_distances,
                        return_distances,
                    )
                )
            elif return_distances:
                selected_distances = selected_candidate_values
                selected_order = np.lexsort(
                    (selected_indices, selected_distances)
                )
                selected_indices = selected_indices[selected_order]
                selected_distances = selected_distances[selected_order]

            nearest_indices[row_number] = selected_indices
            if return_distances:
                nearest_squared_distances[row_number] = selected_distances

        return nearest_squared_distances, nearest_indices

    def _vote_probabilities(self, neighbour_indices):
        probability_one = self.y_[neighbour_indices].sum(axis=1) / self.n_neighbors
        return np.column_stack((1.0 - probability_one, probability_one))

    def _find_neighbours(self, X, return_distances):
        if not hasattr(self, "X_"):
            raise ValueError("fit must be called before prediction or neighbor queries")
        queries = self._validate_X(X)
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("Query feature count differs from training feature count")

        indices = np.empty((len(queries), self.n_neighbors), dtype=np.int64)
        nearest_squared_distances = None
        if return_distances:
            nearest_squared_distances = np.empty_like(indices, dtype=np.float64)

        workspace_rows = min(self.batch_size, len(queries))
        distance_workspace = np.empty(
            (workspace_rows, len(self.X_)),
            dtype=np.float64,
        )

        for start in range(0, len(queries), self.batch_size):
            batch = queries[start:start + self.batch_size]
            with np.errstate(over="raise", invalid="raise"):
                query_squared_norms = np.sum(batch * batch, axis=1)
            if return_distances:
                ranking_values = self._squared_distances(
                    batch,
                    query_squared_norms,
                    distance_workspace,
                )
            else:
                ranking_values = self._ranking_scores(
                    batch,
                    distance_workspace,
                )
            batch_squared_distances, batch_indices = self._select_neighbours(
                ranking_values,
                batch,
                query_squared_norms,
                return_distances,
            )
            indices[start:start + len(batch)] = batch_indices
            if return_distances:
                nearest_squared_distances[start:start + len(batch)] = (
                    batch_squared_distances
                )

        return nearest_squared_distances, indices

    def kneighbors(self, X):
        nearest_squared_distances, indices = self._find_neighbours(
            X,
            return_distances=True,
        )
        distances = np.sqrt(nearest_squared_distances)
        return distances, indices

    def predict_proba(self, X):
        _, neighbour_indices = self._find_neighbours(
            X,
            return_distances=False,
        )
        return self._vote_probabilities(neighbour_indices)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predicted_class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes_[predicted_class_indices]
        return predictions
