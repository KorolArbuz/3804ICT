"""Resident Python scoring adapters; ground-truth test labels are not inputs."""
import time
import hashlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.custom_knn.classifier import CustomKNNClassifier
from .classifier import CustomV2KNNClassifier


class PythonModel:
    def __init__(self, implementation, X_train, y_train, configuration):
        self.implementation = implementation
        self.configuration = configuration
        start = time.perf_counter()
        if implementation == "baseline_python_v5_1":
            self.model = CustomKNNClassifier(n_neighbors=19, batch_size=64)
        elif implementation == "final_python":
            self.model = CustomV2KNNClassifier(101)
        elif implementation == "final_sklearn":
            self.model = KNeighborsClassifier(
                n_neighbors=101,
                weights="distance",
                metric="euclidean",
                algorithm="brute",
                n_jobs=1,
            )
        else:
            raise ValueError("Unknown Python implementation")
        self.model.fit(X_train, y_train)
        self.fit_seconds = time.perf_counter() - start

    def predict(self, X):
        scores = self.model.predict_proba(X)[:, 1]
        threshold = self.configuration["model"]["threshold"]
        labels = (scores > 0.5 if threshold is None else scores >= threshold).astype(
            np.int64
        )
        if (
            scores.ndim != 1
            or len(scores) != len(X)
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("Invalid model scores")
        return scores, labels

    def measured_predict(self, X):
        start = time.perf_counter()
        scores, labels = self.predict(X)
        seconds = time.perf_counter() - start
        # Hash every score and label only after the measured computation has ended.
        digest = hashlib.sha256(
            np.ascontiguousarray(scores, dtype="<f8").tobytes()
            + np.ascontiguousarray(labels, dtype="<i8").tobytes()
        ).hexdigest()
        return (
            {"seconds": seconds, "checksum": "sha256:" + digest, "n_query": len(X)},
            scores,
            labels,
        )
