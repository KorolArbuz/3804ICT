"""Measure prediction scaling with deterministic train and query prefixes."""

import argparse
import csv
import random
import tempfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair, read_json as read_project_json
from src.preprocessing.export import arff_text

from .benchmark_utils import (
    BENCHMARK_DIR,
    DISPLAY_NAMES,
    IMPLEMENTATIONS,
    make_python_models,
    run_cpp_prediction,
    run_weka_prediction,
    timed_python_prediction,
    timestamp,
    write_rows,
)


TRAIN_SIZES = (1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000)
QUERY_SIZES = (100, 250, 500, 1000, 2000, 4000, 6000)
FIELDS = (
    "global_run_order",
    "scaling_dimension",
    "implementation",
    "run_number_within_point",
    "timestamp",
    "runtime_seconds",
    "warmup",
    "train_rows",
    "query_rows",
    "features",
    "k",
    "batch_size",
    "selection_method",
    "queries_per_second",
    "milliseconds_per_query",
    "microseconds_per_query_per_1000_training_rows",
    "prediction_hash",
    "notes",
)


def _write_cpp_features(path, values):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for row in np.asarray(values, dtype=np.float64):
            handle.write(",".join(format(float(value), ".17g") for value in row))
            handle.write("\n")


def _write_cpp_labels(path, labels):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for value in np.asarray(labels):
            handle.write(f"{int(value)}\n")


def _prepare_point(directory, X_train, y_train, X_query, y_query, feature_names):
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "train_features": directory / "train_features.csv",
        "train_labels": directory / "train_labels.csv",
        "test_features": directory / "test_features.csv",
        "test_labels": directory / "test_labels.csv",
    }
    _write_cpp_features(files["train_features"], X_train)
    _write_cpp_labels(files["train_labels"], y_train)
    _write_cpp_features(files["test_features"], X_query)
    _write_cpp_labels(files["test_labels"], y_query)
    train_arff = directory / "train.arff"
    test_arff = directory / "test.arff"
    train_arff.write_text(
        arff_text(X_train, y_train, feature_names), encoding="utf-8", newline="\n"
    )
    test_arff.write_text(
        arff_text(X_query, y_query, feature_names), encoding="utf-8", newline="\n"
    )
    with (directory / "test_ids.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("row_id", "original_id"))
        writer.writerows((index, index) for index in range(len(y_query)))
    return files, train_arff, test_arff


def _row(order, dimension, name, run, warmup, seconds, train_rows, query_rows, digest):
    batch_size = 64 if name == "custom_python_v5_1" else (32 if name == "custom_cpp" else "")
    selection = {
        "custom_cpp": "bounded max-heap",
        "sklearn": "brute",
        "weka": "IBk linear search",
    }.get(name, "exact threshold prefilter plus argpartition")
    seconds = float(seconds)
    return {
        "global_run_order": order,
        "scaling_dimension": dimension,
        "implementation": name,
        "run_number_within_point": run,
        "timestamp": timestamp(),
        "runtime_seconds": f"{seconds:.17g}",
        "warmup": str(bool(warmup)).lower(),
        "train_rows": train_rows,
        "query_rows": query_rows,
        "features": 33,
        "k": 19,
        "batch_size": batch_size,
        "selection_method": selection,
        "queries_per_second": f"{query_rows / seconds:.17g}",
        "milliseconds_per_query": f"{1000 * seconds / query_rows:.17g}",
        "microseconds_per_query_per_1000_training_rows": (
            f"{1_000_000 * seconds / query_rows / (train_rows / 1000):.17g}"
        ),
        "prediction_hash": digest,
        "notes": "Deterministic nested prefix; prediction-only timing; one thread",
    }


def _measure_mode(
    dimension,
    sizes,
    training,
    testing,
    fixed_query_count,
    runs,
    weka_runs,
    seed,
    executable,
    temporary,
    output,
):
    rows = []
    order_number = 0
    randomizer = random.Random(seed)
    feature_names = training["feature_names"].tolist()
    for size in sizes:
        train_rows = size if dimension == "training_rows" else len(training["y"])
        query_rows = fixed_query_count if dimension == "training_rows" else size
        X_train = training["X"][:train_rows]
        y_train = training["y"][:train_rows]
        X_query = testing["X"][:query_rows]
        y_query = testing["y"][:query_rows]
        point_dir = temporary / f"{dimension}_{size}"
        cpp_files, train_arff, test_arff = _prepare_point(
            point_dir, X_train, y_train, X_query, y_query, feature_names
        )
        models = make_python_models(19)
        for model in models.values():
            model.fit(X_train, y_train)
        hashes = {name: set() for name in IMPLEMENTATIONS}

        def measure(name, run, warmup):
            nonlocal order_number
            order_number += 1
            if name in models:
                result = timed_python_prediction(models[name], X_query)
                seconds = result["seconds"]
                digest = result["prediction_hash"]
            elif name == "custom_cpp":
                result = run_cpp_prediction(
                    executable,
                    point_dir,
                    point_dir / f"cpp_{order_number}.json",
                    19,
                    warmups=1,
                    input_files=cpp_files,
                )
                seconds = result["prediction_seconds"][0]
                digest = result["prediction_hash"]
            else:
                result = run_weka_prediction(
                    point_dir / f"weka_{order_number}.csv",
                    19,
                    train=train_arff,
                    test=test_arff,
                )
                seconds = result["prediction_seconds"]
                digest = result["prediction_file_sha256"]
            rows.append(
                _row(
                    order_number,
                    dimension,
                    name,
                    run,
                    warmup,
                    seconds,
                    train_rows,
                    query_rows,
                    digest,
                )
            )
            if not warmup:
                hashes[name].add(digest)

        warmup_order = list(IMPLEMENTATIONS)
        randomizer.shuffle(warmup_order)
        for name in warmup_order:
            measure(name, 0, True)
        for run in range(1, runs + 1):
            names = [name for name in IMPLEMENTATIONS if name != "weka" or run <= weka_runs]
            randomizer.shuffle(names)
            for name in names:
                measure(name, run, False)
        for name, observed in hashes.items():
            expected = runs if name != "weka" else weka_runs
            if expected and len(observed) != 1:
                raise AssertionError(f"{dimension}={size}: {name} output changed")
        if hashes["custom_cpp"] != hashes["custom_python_v5_1"]:
            raise AssertionError(f"{dimension}={size}: C++ and V5.1 differ")
        print(f"Scaling point complete: {dimension}={size}", flush=True)
    write_rows(output, rows, FIELDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--weka-runs", type=int, default=3)
    parser.add_argument("--fixed-query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3804)
    parser.add_argument("--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe"))
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument("--train-output", type=Path, default=BENCHMARK_DIR / "raw_scaling_train.csv")
    parser.add_argument("--query-output", type=Path, default=BENCHMARK_DIR / "raw_scaling_queries.csv")
    args = parser.parse_args(argv)
    if args.runs < 5:
        raise ValueError("Python, scikit-learn, and C++ scaling require at least five runs")
    if not 1 <= args.weka_runs <= args.runs:
        raise ValueError("Weka scaling runs must be between one and the main run count")
    correctness = read_project_json(RESULTS_DIR / "cpp_knn_correctness.json")
    if not correctness.get("passed"):
        raise AssertionError("C++ correctness must pass before scaling")
    training, testing = load_pair(args.train, args.test)
    k = int(read_project_json(RESULTS_DIR / "selected_parameters.json")["selected_k"])
    if (training["X"].shape, testing["X"].shape, k) != ((24000, 33), (6000, 33), 19):
        raise AssertionError("Canonical inputs changed")
    with tempfile.TemporaryDirectory(prefix="knn_scaling_") as directory:
        temporary = Path(directory)
        with threadpool_limits(limits=1):
            _measure_mode(
                "training_rows", TRAIN_SIZES, training, testing,
                args.fixed_query_count, args.runs, args.weka_runs,
                args.seed, args.cpp_executable, temporary, args.train_output,
            )
            _measure_mode(
                "query_rows", QUERY_SIZES, training, testing,
                args.fixed_query_count, args.runs, args.weka_runs,
                args.seed + 1, args.cpp_executable, temporary, args.query_output,
            )
    print(f"Scaling benchmarks -> {args.train_output}, {args.query_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
