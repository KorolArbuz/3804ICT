"""Verify the pure C++ candidate against canonical data and accepted V5.1."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair
from src.custom_knn.classifier import CustomKNNClassifier
from src.evaluation.metrics import classification_metrics

from .evidence import prediction_hash, read_json, sha256_file, write_json


def _load_cpp_inputs(directory):
    directory = Path(directory)
    return {
        "train_X": np.loadtxt(directory / "train_features.csv", delimiter=","),
        "train_y": np.loadtxt(directory / "train_labels.csv", dtype=np.int64),
        "test_X": np.loadtxt(directory / "test_features.csv", delimiter=","),
        "test_y": np.loadtxt(directory / "test_labels.csv", dtype=np.int64),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a C++ KNN result with canonical NPZ data and V5.1."
    )
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument("--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp")
    parser.add_argument("--cpp-result", type=Path, required=True)
    parser.add_argument("--k", type=int, default=19)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "cpp_knn_correctness.json"
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        raise ValueError("k must be at least one")

    training, testing = load_pair(args.train, args.test)
    cpp_inputs = _load_cpp_inputs(args.cpp_input_dir)
    cpp = read_json(args.cpp_result)

    canonical_checks = {
        "train_features_exact": bool(np.array_equal(cpp_inputs["train_X"], training["X"])),
        "train_labels_exact": bool(np.array_equal(cpp_inputs["train_y"], training["y"])),
        "test_features_exact": bool(np.array_equal(cpp_inputs["test_X"], testing["X"])),
        "test_labels_exact": bool(np.array_equal(cpp_inputs["test_y"], testing["y"])),
        "feature_order_exact": bool(
            np.array_equal(training["feature_names"], testing["feature_names"])
        ),
        "id_excluded": not any(
            str(name).lower() in {"id", "target", "class"}
            for name in training["feature_names"]
        ),
        "all_values_finite": bool(
            np.isfinite(training["X"]).all() and np.isfinite(testing["X"]).all()
        ),
    }
    if not all(canonical_checks.values()):
        raise AssertionError(f"Canonical C++ input check failed: {canonical_checks}")

    classifier = CustomKNNClassifier(args.k)
    with threadpool_limits(limits=1):
        classifier.fit(training["X"], training["y"])
        probabilities = classifier.predict_proba(testing["X"])
    accepted_labels = classifier.classes_[np.argmax(probabilities, axis=1)]
    accepted_votes = np.rint(probabilities[:, 1] * args.k).astype(np.int64)

    cpp_labels = np.asarray(cpp["predicted_labels"], dtype=np.int64)
    cpp_votes = np.asarray(cpp["positive_vote_counts"], dtype=np.int64)
    cpp_fractions = np.asarray(cpp["positive_vote_fractions"], dtype=np.float64)
    accepted_hash = prediction_hash(accepted_labels, accepted_votes)
    labels_equal = cpp_labels == accepted_labels
    votes_equal = cpp_votes == accepted_votes
    fractions_equal = cpp_fractions == probabilities[:, 1]
    metrics = classification_metrics(testing["y"], cpp_labels, cpp_fractions)

    result = {
        "implementation": "custom_cpp_exact_knn",
        "candidate_status": "experimental",
        "canonical_inputs": {
            "train_rows": int(training["X"].shape[0]),
            "test_rows": int(testing["X"].shape[0]),
            "feature_count": int(training["X"].shape[1]),
            "checks": canonical_checks,
            "manifest": read_json(args.cpp_input_dir / "manifest.json"),
        },
        "accepted_v5_1": {
            "source": "src/custom_knn/classifier.py",
            "source_sha256": sha256_file("src/custom_knn/classifier.py"),
            "prediction_hash": accepted_hash,
        },
        "cpp": {
            "build_mode": cpp["build_mode"],
            "compiler": cpp["compiler"],
            "prediction_hash": cpp["prediction_hash"],
        },
        "equivalence": {
            "prediction_matches": int(labels_equal.sum()),
            "prediction_count": int(len(labels_equal)),
            "positive_vote_count_matches": int(votes_equal.sum()),
            "positive_vote_fraction_matches": int(fractions_equal.sum()),
            "max_absolute_vote_fraction_difference": float(
                np.max(np.abs(cpp_fractions - probabilities[:, 1]))
            ),
            "prediction_hash_equal": cpp["prediction_hash"] == accepted_hash,
            "cpp_repeated_run_check_exercised": cpp["measured_run_count"] >= 2,
        },
        "classification_metrics": metrics,
        "confusion_matrix": cpp["confusion_matrix"],
    }
    required = result["equivalence"]
    if not (
        required["prediction_matches"] == required["prediction_count"] == 6000
        and required["positive_vote_count_matches"] == 6000
        and required["positive_vote_fraction_matches"] == 6000
        and required["prediction_hash_equal"]
    ):
        raise AssertionError(f"C++/V5.1 equivalence failed: {required}")
    result["passed"] = True
    write_json(args.output, result)
    print(
        f"C++ equivalence passed: 6000/6000 labels and votes; "
        f"hash {accepted_hash} -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
