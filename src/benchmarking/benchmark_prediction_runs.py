"""Collect the comprehensive controlled prediction-only benchmark."""

import argparse
import random
import tempfile
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair, read_json as read_project_json
from src.cpp_knn.evidence import sha256_file

from .benchmark_utils import (
    BENCHMARK_DIR,
    DISPLAY_NAMES,
    IMPLEMENTATIONS,
    make_python_models,
    read_json,
    run_cpp_prediction,
    run_weka_prediction,
    timed_python_prediction,
    timestamp,
    write_rows,
)


FIELDS = (
    "global_run_order",
    "implementation",
    "implementation_version",
    "run_number_within_implementation",
    "timestamp",
    "runtime_seconds",
    "warmup",
    "train_rows",
    "test_rows",
    "features",
    "k",
    "batch_size",
    "selection_method",
    "fit_seconds_outside_prediction",
    "thread_count",
    "timing_scope",
    "prediction_hash",
    "compiler",
    "build_type",
    "executable_sha256",
    "cpu_frequency_mhz",
    "temperature_celsius",
    "notes",
)

VERSIONS = {
    "custom_python_v5_1": "V5.1 accepted",
    "sklearn": "scikit-learn 1.2.1",
    "weka": "Weka 3.8.6 IBk",
    "custom_cpp": "C++20 experimental",
}


def _row(order, run, name, warmup, runtime, fit, digest, metadata, shape, k):
    batch_size = 64 if name == "custom_python_v5_1" else ""
    selection = ""
    compiler = ""
    build_type = ""
    executable_sha256 = ""
    if name == "custom_cpp":
        batch_size = metadata["batch_size"]
        selection = metadata["selection"]
        compiler = metadata["compiler"]
        build_type = metadata["build_mode"]
        executable_sha256 = metadata["executable_sha256"]
    elif name == "sklearn":
        selection = "brute"
    elif name == "weka":
        selection = "IBk linear search"
    return {
        "global_run_order": order,
        "implementation": name,
        "implementation_version": VERSIONS[name],
        "run_number_within_implementation": run,
        "timestamp": timestamp(),
        "runtime_seconds": f"{float(runtime):.17g}",
        "warmup": str(bool(warmup)).lower(),
        "train_rows": shape[0],
        "test_rows": shape[1],
        "features": shape[2],
        "k": k,
        "batch_size": batch_size,
        "selection_method": selection,
        "fit_seconds_outside_prediction": f"{float(fit):.17g}",
        "thread_count": 1,
        "timing_scope": metadata["timing_scope"],
        "prediction_hash": digest,
        "compiler": compiler,
        "build_type": build_type,
        "executable_sha256": executable_sha256,
        "cpu_frequency_mhz": "",
        "temperature_celsius": "",
        "notes": metadata["notes"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe")
    )
    parser.add_argument(
        "--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp"
    )
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument("--seed", type=int, default=3804)
    parser.add_argument(
        "--output", type=Path, default=BENCHMARK_DIR / "raw_prediction_runs.csv"
    )
    args = parser.parse_args(argv)
    if args.runs < 15:
        raise ValueError("At least 15 controlled prediction runs are required")
    correctness = read_json(RESULTS_DIR / "cpp_knn_correctness.json")
    if not correctness.get("passed"):
        raise AssertionError("The C++ real-data verifier must pass before timing")

    training, testing = load_pair(args.train, args.test)
    k = int(read_project_json(RESULTS_DIR / "selected_parameters.json")["selected_k"])
    if (training["X"].shape, testing["X"].shape, k) != ((24000, 33), (6000, 33), 19):
        raise AssertionError("Canonical prepared data or selected k changed")
    shape = (len(training["y"]), len(testing["y"]), training["X"].shape[1])

    models = make_python_models(k)
    fit_seconds = {}
    rows = []
    order_number = 0
    timed_hashes = {name: set() for name in IMPLEMENTATIONS}
    randomizer = random.Random(args.seed)
    executable_hash = sha256_file(args.cpp_executable)

    with tempfile.TemporaryDirectory(prefix="knn_controlled_prediction_") as directory:
        temporary = Path(directory)
        with threadpool_limits(limits=1):
            for name, model in models.items():
                start = perf_counter()
                model.fit(training["X"], training["y"])
                fit_seconds[name] = perf_counter() - start

            def measure(name, run, warmup):
                nonlocal order_number
                order_number += 1
                if name in models:
                    result = timed_python_prediction(models[name], testing["X"])
                    metadata = {
                        "timing_scope": "predict_proba plus class argmax; prepared arrays in memory",
                        "notes": "Numerical thread pools limited to one thread",
                    }
                    runtime = result["seconds"]
                    fit = fit_seconds[name]
                    digest = result["prediction_hash"]
                elif name == "custom_cpp":
                    result = run_cpp_prediction(
                        args.cpp_executable,
                        args.cpp_input_dir,
                        temporary / f"cpp_{order_number}.json",
                        k,
                        warmups=1,
                    )
                    result["executable_sha256"] = executable_hash
                    result["notes"] = "Fresh process with an internal untimed warm-up"
                    metadata = result
                    runtime = result["prediction_seconds"][0]
                    fit = result["fit_seconds"]
                    digest = result["prediction_hash"]
                else:
                    result = run_weka_prediction(temporary / f"weka_{order_number}.csv", k)
                    metadata = {
                        "timing_scope": result["timing_scope"],
                        "notes": "Fresh JVM; internal timer excludes JVM startup, loading, fit, and output",
                    }
                    runtime = result["prediction_seconds"]
                    fit = result["fit_seconds"]
                    digest = result["prediction_file_sha256"]
                rows.append(
                    _row(order_number, run, name, warmup, runtime, fit, digest, metadata, shape, k)
                )
                if not warmup:
                    timed_hashes[name].add(digest)

            warmup_order = list(IMPLEMENTATIONS)
            randomizer.shuffle(warmup_order)
            for name in warmup_order:
                measure(name, 0, True)
            for run in range(1, args.runs + 1):
                trial_order = list(IMPLEMENTATIONS)
                randomizer.shuffle(trial_order)
                for name in trial_order:
                    measure(name, run, False)

    for name, hashes in timed_hashes.items():
        if len(hashes) != 1:
            raise AssertionError(f"{name} output changed between controlled runs")
    if timed_hashes["custom_cpp"] != timed_hashes["custom_python_v5_1"]:
        raise AssertionError("C++ no longer exactly matches accepted V5.1")
    write_rows(args.output, rows, FIELDS)
    print(f"Controlled prediction benchmark: {len(rows) - 4} timed rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
