"""Measure complete per-implementation runs from prepared inputs through metrics."""

import argparse
import random
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_npz, read_json as read_project_json
from src.custom_knn.runner import run as run_custom
from src.evaluation.metrics import classification_metrics, evaluate_file
from src.sklearn_knn.runner import run as run_sklearn

from .benchmark_utils import (
    BENCHMARK_DIR,
    DISPLAY_NAMES,
    IMPLEMENTATIONS,
    read_json,
    run_cpp_prediction,
    run_weka_prediction,
    timestamp,
    write_json,
    write_rows,
)


FIELDS = (
    "sequence_index",
    "trial_index",
    "timestamp",
    "implementation",
    "display_name",
    "total_seconds",
    "fit_seconds",
    "prediction_seconds",
    "metric_seconds",
    "loading_export_startup_overhead_seconds",
    "measurement_type",
    "notes",
)


def _python_pipeline(name, temporary, trial, train, test, selected_parameters):
    output = temporary / f"full_{trial}_{name}.csv"
    runner = run_custom if name == "custom_python_v5_1" else run_sklearn
    start = perf_counter()
    runtime = runner(
        train,
        test,
        output,
        k=None,
        selected_parameters=selected_parameters,
    )
    metric_start = perf_counter()
    evaluate_file(output, test, "custom" if name == "custom_python_v5_1" else name)
    metric_seconds = perf_counter() - metric_start
    total_seconds = perf_counter() - start
    return total_seconds, runtime, metric_seconds


def _weka_pipeline(temporary, trial, test, k):
    output = temporary / f"full_{trial}_weka.csv"
    start = perf_counter()
    runtime = run_weka_prediction(output, k)
    metric_start = perf_counter()
    evaluate_file(output, test, "weka")
    metric_seconds = perf_counter() - metric_start
    total_seconds = perf_counter() - start
    return total_seconds, runtime, metric_seconds


def _cpp_pipeline(executable, cpp_input_dir, temporary, trial, test, k):
    output = temporary / f"full_{trial}_cpp.json"
    start = perf_counter()
    runtime = run_cpp_prediction(
        executable, cpp_input_dir, output, k, warmups=0
    )
    metric_start = perf_counter()
    y_true = load_npz(test)["y"]
    labels = np.asarray(runtime["predicted_labels"], dtype=np.int64)
    probabilities = np.asarray(runtime["positive_vote_fractions"], dtype=float)
    classification_metrics(y_true, labels, probabilities)
    metric_seconds = perf_counter() - metric_start
    total_seconds = perf_counter() - start
    return total_seconds, runtime, metric_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe")
    )
    parser.add_argument(
        "--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp"
    )
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument(
        "--selected-parameters",
        type=Path,
        default=RESULTS_DIR / "selected_parameters.json",
    )
    parser.add_argument(
        "--output", type=Path, default=BENCHMARK_DIR / "raw_full_pipeline_benchmark.csv"
    )
    parser.add_argument("--seed", type=int, default=3804)
    args = parser.parse_args(argv)
    if args.runs < 1:
        raise ValueError("Full-pipeline run count must be at least one")
    correctness = read_json(RESULTS_DIR / "cpp_knn_correctness.json")
    if not correctness.get("passed"):
        raise AssertionError("C++ real-data verifier must pass before benchmarking")
    k = int(read_project_json(args.selected_parameters)["selected_k"])

    rows = []
    sequence = 0
    randomizer = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix="knn_full_pipeline_") as directory:
        temporary = Path(directory)
        with threadpool_limits(limits=1):
            for trial in range(1, args.runs + 1):
                order = list(IMPLEMENTATIONS)
                randomizer.shuffle(order)
                for name in order:
                    sequence += 1
                    if name in {"custom_python_v5_1", "sklearn"}:
                        total, runtime, metric = _python_pipeline(
                            name,
                            temporary,
                            trial,
                            args.train,
                            args.test,
                            args.selected_parameters,
                        )
                    elif name == "weka":
                        total, runtime, metric = _weka_pipeline(
                            temporary, trial, args.test, k
                        )
                    else:
                        total, runtime, metric = _cpp_pipeline(
                            args.cpp_executable,
                            args.cpp_input_dir,
                            temporary,
                            trial,
                            args.test,
                            k,
                        )
                    fit = float(runtime["fit_seconds"])
                    prediction = (
                        float(runtime["prediction_seconds"][0])
                        if isinstance(runtime["prediction_seconds"], list)
                        else float(runtime["prediction_seconds"])
                    )
                    overhead = max(0.0, total - fit - prediction - metric)
                    rows.append({
                        "sequence_index": sequence,
                        "trial_index": trial,
                        "timestamp": timestamp(),
                        "implementation": name,
                        "display_name": DISPLAY_NAMES[name],
                        "total_seconds": f"{total:.17g}",
                        "fit_seconds": f"{fit:.17g}",
                        "prediction_seconds": f"{prediction:.17g}",
                        "metric_seconds": f"{metric:.17g}",
                        "loading_export_startup_overhead_seconds": f"{overhead:.17g}",
                        "measurement_type": "full prepared-input pipeline",
                        "notes": (
                            "Includes prepared-file loading, model construction/fit, prediction, "
                            "artifact writing, and metric generation; shared raw-data "
                            "preprocessing and k selection are fixed and excluded."
                        ),
                    })

    write_rows(args.output, rows, FIELDS)
    write_json(args.output.with_suffix(".json"), rows)
    medians = {}
    for name in IMPLEMENTATIONS:
        values = [
            float(row["total_seconds"])
            for row in rows
            if row["implementation"] == name
        ]
        medians[name] = float(np.median(values))
    print(
        "Full-pipeline benchmark complete: "
        + ", ".join(f"{name}={value:.3f}s" for name, value in medians.items()),
        flush=True,
    )
    print(f"Raw rows -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

