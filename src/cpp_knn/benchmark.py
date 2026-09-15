"""Interleaved single-thread benchmark for C++, accepted V5.1, and sklearn."""

import argparse
import csv
import importlib.metadata
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_info, threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair, read_json as read_project_json
from src.custom_knn.classifier import CustomKNNClassifier

from .evidence import prediction_hash, read_json, sha256_file, timing_summary, write_json


IMPLEMENTATIONS = ("custom_cpp", "custom_python_v5_1", "sklearn")


def _cpu_name():
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except (OSError, ImportError):
            pass
    return platform.processor() or platform.machine()


def _command_version(command):
    try:
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    text = (completed.stdout + completed.stderr).strip()
    return text.splitlines()[0] if text else "unavailable"


def _power_mode():
    if sys.platform != "win32":
        return "not queried on this operating system"
    return _command_version(["powercfg", "/getactivescheme"])


def _weka_version():
    text = Path("weka/pom.xml").read_text(encoding="utf-8")
    found = re.search(
        r"<artifactId>weka-stable</artifactId>\s*<version>([^<]+)</version>", text
    )
    return found.group(1) if found else "not declared"


def _existing_weka_result():
    path = RESULTS_DIR / "runtime_comparison.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["implementation"] == "weka")
    return {
        "source": str(path).replace("\\", "/"),
        "fit_seconds": float(row["fit_seconds"]),
        "prediction_seconds": float(row["prediction_seconds"]),
        "measurement_note": (
            "Existing genuine Java/Weka IBk full-pipeline result; a single model call, "
            "shown for context and excluded from repeated-run ratios."
        ),
    }


def _run_cpp(executable, input_dir, output, k, batch_size, warmups, runs, selection):
    command = [
        str(executable),
        "--train-features", str(input_dir / "train_features.csv"),
        "--train-labels", str(input_dir / "train_labels.csv"),
        "--test-features", str(input_dir / "test_features.csv"),
        "--test-labels", str(input_dir / "test_labels.csv"),
        "--k", str(k),
        "--output", str(output),
        "--warmups", str(warmups),
        "--runs", str(runs),
        "--batch-size", str(batch_size),
        "--selection", selection,
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    return read_json(output)


def _timed_python_prediction(classifier, queries):
    start = perf_counter()
    probabilities = classifier.predict_proba(queries)
    labels = classifier.classes_[np.argmax(probabilities, axis=1)]
    seconds = perf_counter() - start
    votes = np.rint(probabilities[:, 1] * classifier.n_neighbors).astype(np.int64)
    return seconds, labels, votes


def _screen_cpp(executable, input_dir, temporary_dir, k):
    candidates = (
        ("heap", 16),
        ("heap", 32),
        ("heap", 64),
        ("nth", 32),
    )
    results = {}
    for selection, batch_size in candidates:
        key = f"{selection}_batch_{batch_size}"
        result = _run_cpp(
            executable,
            input_dir,
            temporary_dir / f"screen_{key}.json",
            k,
            batch_size,
            warmups=1,
            runs=3,
            selection=selection,
        )
        results[key] = timing_summary(result["prediction_seconds"])
    selected = min(results, key=lambda key: results[key]["median_seconds"])
    if selected != "heap_batch_32":
        raise RuntimeError(
            f"The documented heap/batch-32 configuration did not win the bounded screen: {selected}"
        )
    return {
        "candidate_policy": "heap vs nth_element at batch 32; heap batches 16, 32, 64",
        "warmups_each": 1,
        "measured_runs_each": 3,
        "candidates": results,
        "selected": selected,
    }


def _git_value(*arguments):
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=C:/Users/Denis/Desktop/3804ICT", *arguments],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_executable = Path("build-cpp-native") / (
        "cpp_knn.exe" if os.name == "nt" else "cpp_knn"
    )
    parser.add_argument("--executable", type=Path, default=default_executable)
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument("--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp")
    parser.add_argument(
        "--selected-parameters",
        type=Path,
        default=RESULTS_DIR / "selected_parameters.json",
    )
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "cpp_knn_benchmark.json"
    )
    args = parser.parse_args(argv)
    if args.runs < 15:
        raise ValueError("The final protocol requires at least 15 measured runs")
    if not args.executable.is_file():
        raise FileNotFoundError(f"Build the C++ executable first: {args.executable}")

    training, testing = load_pair(args.train, args.test)
    k = int(read_project_json(args.selected_parameters)["selected_k"])
    if (training["X"].shape, testing["X"].shape, k) != ((24000, 33), (6000, 33), 19):
        raise AssertionError("Canonical dimensions or selected k differ from verified protocol")

    custom = CustomKNNClassifier(k)
    sklearn = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
        n_jobs=1,
    )
    fit_seconds = {}
    timings = {name: [] for name in IMPLEMENTATIONS}
    final_outputs = {}
    trial_orders = []
    cpp_metadata = None
    measurement_threadpools = None
    randomizer = random.Random(3804)

    with tempfile.TemporaryDirectory(prefix="cpp_knn_benchmark_") as directory:
        temporary_dir = Path(directory)
        screen = _screen_cpp(args.executable, args.cpp_input_dir, temporary_dir, k)

        with threadpool_limits(limits=1):
            measurement_threadpools = threadpool_info()
            start = perf_counter()
            custom.fit(training["X"], training["y"])
            fit_seconds["custom_python_v5_1"] = perf_counter() - start
            start = perf_counter()
            sklearn.fit(training["X"], training["y"])
            fit_seconds["sklearn"] = perf_counter() - start

            # Each Python model is warmed once. Each C++ subprocess receives its
            # own internal warm-up because process-local code/data caches reset.
            _timed_python_prediction(custom, testing["X"])
            _timed_python_prediction(sklearn, testing["X"])

            for trial in range(args.runs):
                order = list(IMPLEMENTATIONS)
                randomizer.shuffle(order)
                trial_orders.append(order)
                for name in order:
                    if name == "custom_cpp":
                        cpp_metadata = _run_cpp(
                            args.executable,
                            args.cpp_input_dir,
                            temporary_dir / f"trial_{trial}.json",
                            k,
                            batch_size=32,
                            warmups=1,
                            runs=1,
                            selection="heap",
                        )
                        seconds = float(cpp_metadata["prediction_seconds"][0])
                        labels = np.asarray(cpp_metadata["predicted_labels"], dtype=np.int64)
                        votes = np.asarray(
                            cpp_metadata["positive_vote_counts"], dtype=np.int64
                        )
                    elif name == "custom_python_v5_1":
                        seconds, labels, votes = _timed_python_prediction(
                            custom, testing["X"]
                        )
                    else:
                        seconds, labels, votes = _timed_python_prediction(
                            sklearn, testing["X"]
                        )
                    timings[name].append(seconds)
                    digest = prediction_hash(labels, votes)
                    if name == "custom_cpp" and digest != cpp_metadata["prediction_hash"]:
                        raise AssertionError("C++ and Python hash implementations disagree")
                    if name in final_outputs and digest != final_outputs[name]["prediction_hash"]:
                        raise AssertionError(f"{name} predictions changed between trials")
                    final_outputs[name] = {
                        "prediction_hash": digest,
                        "labels": labels,
                        "votes": votes,
                    }

    if cpp_metadata is None:
        raise AssertionError("C++ benchmark did not run")
    custom_equal = np.array_equal(
        final_outputs["custom_cpp"]["labels"],
        final_outputs["custom_python_v5_1"]["labels"],
    ) and np.array_equal(
        final_outputs["custom_cpp"]["votes"],
        final_outputs["custom_python_v5_1"]["votes"],
    )
    if not custom_equal:
        raise AssertionError("C++ output does not match accepted V5.1")

    summaries = {name: timing_summary(values) for name, values in timings.items()}
    cpp_median = summaries["custom_cpp"]["median_seconds"]
    custom_median = summaries["custom_python_v5_1"]["median_seconds"]
    sklearn_median = summaries["sklearn"]["median_seconds"]
    paired_v5_over_cpp = [
        custom_seconds / cpp_seconds
        for custom_seconds, cpp_seconds in zip(
            timings["custom_python_v5_1"], timings["custom_cpp"]
        )
    ]
    paired_cpp_over_sklearn = [
        cpp_seconds / sklearn_seconds
        for cpp_seconds, sklearn_seconds in zip(
            timings["custom_cpp"], timings["sklearn"]
        )
    ]
    result = {
        "artifact": "pure C++20 exact KNN final benchmark",
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "source_sha256": {
            path: sha256_file(path)
            for path in (
                "CMakeLists.txt",
                "src/cpp_knn/knn.hpp",
                "src/cpp_knn/knn.cpp",
                "src/cpp_knn/data_io.hpp",
                "src/cpp_knn/data_io.cpp",
                "src/cpp_knn/main.cpp",
                "src/custom_knn/classifier.py",
                "src/sklearn_knn/runner.py",
                "weka/src/main/java/project/WekaIBkRunner.java",
            )
        },
        "protocol": {
            "training_rows": 24000,
            "test_rows": 6000,
            "feature_count": 33,
            "k": k,
            "metric": "exact exhaustive squared Euclidean distance",
            "voting": "uniform",
            "thread_count": 1,
            "warmup": (
                "one call per Python model; one internal call in every fresh C++ process"
            ),
            "measured_runs_each": args.runs,
            "interleaving_seed": 3804,
            "trial_orders": trial_orders,
            "timing_scope": (
                "predict_proba plus argmax for Python; exact distance, selection, voting, "
                "and output vectors for C++; fit, input parsing, process startup, and "
                "serialization excluded"
            ),
        },
        "environment": {
            "cpu": _cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "threadpoolctl": importlib.metadata.version("threadpoolctl"),
            "java": _command_version(["java", "-version"]),
            "weka": _weka_version(),
            "cpp_compiler": cpp_metadata["compiler"],
            "cpp_build_mode": cpp_metadata["build_mode"],
            "power_mode": _power_mode(),
            "threadpools_during_measurement": measurement_threadpools,
            "frequency_limitation": (
                "CPU frequency, background load, and thermal state were not locked; "
                "randomized interleaving reduces but cannot remove this source of variation."
            ),
        },
        "configuration_screen": screen,
        "implementations": {
            "custom_cpp": {
                **summaries["custom_cpp"],
                "fit_seconds": cpp_metadata["fit_seconds"],
                "prediction_hash": final_outputs["custom_cpp"]["prediction_hash"],
                "compiler": cpp_metadata["compiler"],
                "build_mode": cpp_metadata["build_mode"],
                "batch_size": 32,
                "selection": "bounded max-heap",
                "peak_auxiliary_bytes_estimate": cpp_metadata[
                    "peak_auxiliary_bytes_estimate"
                ],
            },
            "custom_python_v5_1": {
                **summaries["custom_python_v5_1"],
                "fit_seconds": fit_seconds["custom_python_v5_1"],
                "prediction_hash": final_outputs["custom_python_v5_1"][
                    "prediction_hash"
                ],
                "source_sha256": sha256_file("src/custom_knn/classifier.py"),
            },
            "sklearn": {
                **summaries["sklearn"],
                "fit_seconds": fit_seconds["sklearn"],
                "prediction_hash": final_outputs["sklearn"]["prediction_hash"],
                "configuration": (
                    "KNeighborsClassifier(brute, euclidean, uniform, n_jobs=1)"
                ),
            },
            "weka": _existing_weka_result(),
        },
        "correctness": {
            "cpp_vs_v5_1_predictions": "6000/6000",
            "cpp_vs_v5_1_positive_vote_counts": "6000/6000 exact",
            "cpp_vs_v5_1_prediction_hash_equal": (
                final_outputs["custom_cpp"]["prediction_hash"]
                == final_outputs["custom_python_v5_1"]["prediction_hash"]
            ),
        },
        "ratios": {
            "v5_1_median_over_cpp_median": custom_median / cpp_median,
            "cpp_median_over_sklearn_median": cpp_median / sklearn_median,
            "paired_v5_1_over_cpp": paired_v5_over_cpp,
            "median_paired_v5_1_over_cpp": timing_summary(paired_v5_over_cpp)[
                "median_seconds"
            ],
            "paired_cpp_over_sklearn": paired_cpp_over_sklearn,
            "median_paired_cpp_over_sklearn": timing_summary(
                paired_cpp_over_sklearn
            )["median_seconds"],
            "cpp_faster_than_v5_1_trial_count": sum(
                cpp_seconds < custom_seconds
                for cpp_seconds, custom_seconds in zip(
                    timings["custom_cpp"], timings["custom_python_v5_1"]
                )
            ),
        },
        "performance_gates": {
            "cpp_at_least_10_percent_faster_than_v5_1": cpp_median <= 0.9 * custom_median,
            "cpp_within_1_10x_sklearn": cpp_median <= 1.1 * sklearn_median,
        },
    }
    write_json(args.output, result)
    print(
        f"Final medians: C++ {cpp_median:.6f}s; V5.1 {custom_median:.6f}s; "
        f"sklearn {sklearn_median:.6f}s -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
