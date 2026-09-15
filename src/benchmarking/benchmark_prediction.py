"""Collect interleaved prediction-only timings for all runnable implementations."""

import argparse
import random
import tempfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair, read_json as read_project_json

from .benchmark_utils import (
    BENCHMARK_DIR,
    DISPLAY_NAMES,
    IMPLEMENTATIONS,
    collect_environment,
    run_cpp_prediction,
    run_weka_prediction,
    make_python_models,
    timed_python_prediction,
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
    "is_warmup",
    "prediction_seconds",
    "fit_seconds_outside_prediction",
    "prediction_hash",
    "thread_count",
    "timing_scope",
    "notes",
)


def _row(sequence, trial, name, warmup, seconds, fit_seconds, digest, scope, notes):
    return {
        "sequence_index": sequence,
        "trial_index": trial,
        "timestamp": timestamp(),
        "implementation": name,
        "display_name": DISPLAY_NAMES[name],
        "is_warmup": str(bool(warmup)).lower(),
        "prediction_seconds": f"{float(seconds):.17g}",
        "fit_seconds_outside_prediction": f"{float(fit_seconds):.17g}",
        "prediction_hash": digest,
        "thread_count": 1,
        "timing_scope": scope,
        "notes": notes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument(
        "--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe")
    )
    parser.add_argument(
        "--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp"
    )
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument(
        "--output", type=Path, default=BENCHMARK_DIR / "raw_prediction_benchmark.csv"
    )
    parser.add_argument("--seed", type=int, default=3804)
    args = parser.parse_args(argv)
    if args.runs < 15:
        raise ValueError("Prediction-only reporting requires at least 15 timed runs")
    required = (
        args.cpp_executable,
        args.cpp_input_dir / "manifest.json",
        Path("weka/target/knn-weka-runner.jar"),
        PROCESSED_DATA_DIR / "train.arff",
        PROCESSED_DATA_DIR / "test.arff",
    )
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Required benchmark inputs are missing: {missing}")

    training, testing = load_pair(args.train, args.test)
    k = int(read_project_json(RESULTS_DIR / "selected_parameters.json")["selected_k"])
    if (training["X"].shape, testing["X"].shape, k) != ((24000, 33), (6000, 33), 19):
        raise AssertionError("Prepared data dimensions or selected k changed")

    models = make_python_models(k)
    fit_seconds = {}
    rows = []
    sequence = 0
    observed_hashes = {name: set() for name in IMPLEMENTATIONS}
    cpp_metadata = None
    randomizer = random.Random(args.seed)

    with tempfile.TemporaryDirectory(prefix="knn_prediction_benchmark_") as directory:
        temporary = Path(directory)
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            for name, model in models.items():
                from time import perf_counter

                start = perf_counter()
                model.fit(training["X"], training["y"])
                fit_seconds[name] = perf_counter() - start

            # Explicit warm-up records are retained but excluded from summaries.
            for name in IMPLEMENTATIONS:
                sequence += 1
                if name in models:
                    result = timed_python_prediction(models[name], testing["X"])
                    seconds = result["seconds"]
                    digest = result["prediction_hash"]
                    fit_value = fit_seconds[name]
                    scope = "predict_proba plus argmax; prepared arrays already in memory"
                    notes = "Explicit untimed warm-up pass"
                elif name == "custom_cpp":
                    cpp_metadata = run_cpp_prediction(
                        args.cpp_executable,
                        args.cpp_input_dir,
                        temporary / "warmup_cpp.json",
                        k,
                        warmups=1,
                    )
                    seconds = cpp_metadata["prediction_seconds"][0]
                    digest = cpp_metadata["prediction_hash"]
                    fit_value = cpp_metadata["fit_seconds"]
                    scope = cpp_metadata["timing_scope"]
                    notes = "Explicit warm-up command; includes an additional internal warm-up"
                else:
                    runtime = run_weka_prediction(temporary / "warmup_weka.csv", k)
                    seconds = runtime["prediction_seconds"]
                    digest = runtime["prediction_file_sha256"]
                    fit_value = runtime["fit_seconds"]
                    scope = runtime["timing_scope"]
                    notes = "Explicit warm-up JVM; each timed Weka trial uses a fresh JVM"
                rows.append(
                    _row(
                        sequence, 0, name, True, seconds, fit_value, digest, scope, notes
                    )
                )

            for trial in range(1, args.runs + 1):
                order = list(IMPLEMENTATIONS)
                randomizer.shuffle(order)
                for name in order:
                    sequence += 1
                    if name in models:
                        result = timed_python_prediction(models[name], testing["X"])
                        seconds = result["seconds"]
                        digest = result["prediction_hash"]
                        fit_value = fit_seconds[name]
                        scope = "predict_proba plus argmax; prepared arrays already in memory"
                        notes = "Accepted current implementation; Python numerical pools limited to one"
                    elif name == "custom_cpp":
                        cpp_metadata = run_cpp_prediction(
                            args.cpp_executable,
                            args.cpp_input_dir,
                            temporary / f"trial_{trial}_cpp.json",
                            k,
                            warmups=1,
                        )
                        seconds = cpp_metadata["prediction_seconds"][0]
                        digest = cpp_metadata["prediction_hash"]
                        fit_value = cpp_metadata["fit_seconds"]
                        scope = cpp_metadata["timing_scope"]
                        notes = "Fresh process with one internal untimed warm-up; native Release"
                    else:
                        runtime = run_weka_prediction(
                            temporary / f"trial_{trial}_weka.csv", k
                        )
                        seconds = runtime["prediction_seconds"]
                        digest = runtime["prediction_file_sha256"]
                        fit_value = runtime["fit_seconds"]
                        scope = runtime["timing_scope"]
                        notes = "Fresh one-CPU JVM; Java timer excludes loading, fit, output, and startup"
                    observed_hashes[name].add(digest)
                    rows.append(
                        _row(
                            sequence,
                            trial,
                            name,
                            False,
                            seconds,
                            fit_value,
                            digest,
                            scope,
                            notes,
                        )
                    )

    if cpp_metadata is None:
        raise AssertionError("C++ candidate was not measured")
    for name, hashes in observed_hashes.items():
        if len(hashes) != 1:
            raise AssertionError(f"{name} output changed between timed runs")
    if observed_hashes["custom_cpp"] != observed_hashes["custom_python_v5_1"]:
        raise AssertionError("C++ no longer matches accepted V5.1 predictions and votes")

    write_rows(args.output, rows, FIELDS)
    write_json(args.output.with_suffix(".json"), rows)
    environment = collect_environment(pools, cpp_metadata)
    environment["protocol"] = {
        "runs_each": args.runs,
        "warmups_each": 1,
        "random_seed": args.seed,
        "implementations": list(IMPLEMENTATIONS),
        "training_shape": list(training["X"].shape),
        "test_shape": list(testing["X"].shape),
        "selected_k": k,
        "custom_python_batch_size": models["custom_python_v5_1"].batch_size,
        "cpp_batch_size": cpp_metadata["batch_size"],
    }
    write_json(BENCHMARK_DIR / "environment.json", environment)
    medians = {}
    for name in IMPLEMENTATIONS:
        values = [
            float(row["prediction_seconds"])
            for row in rows
            if row["implementation"] == name and row["is_warmup"] == "false"
        ]
        medians[name] = float(np.median(values))
    print(
        "Prediction benchmark complete: "
        + ", ".join(f"{name}={value:.3f}s" for name, value in medians.items()),
        flush=True,
    )
    print(f"Raw rows -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

