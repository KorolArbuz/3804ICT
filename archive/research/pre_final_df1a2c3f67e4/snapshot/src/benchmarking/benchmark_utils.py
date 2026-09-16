"""Shared helpers for runtime benchmark collection and reporting."""

import csv
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from . import SINGLE_THREAD_ENVIRONMENT

from src.common.config import DEFAULT_BATCH_SIZE, PROCESSED_DATA_DIR, RESULTS_DIR, WEKA_DIR
from src.cpp_knn.evidence import prediction_hash, sha256_file
from src.custom_knn.classifier import CustomKNNClassifier


BENCHMARK_DIR = RESULTS_DIR / "runtime_benchmarks"
FIGURES_DIR = BENCHMARK_DIR / "figures"
IMPLEMENTATIONS = ("custom_python_v5_1", "sklearn", "weka", "custom_cpp")
DISPLAY_NAMES = {
    "custom_python_v5_1": "Custom Python KNN V5.1",
    "sklearn": "scikit-learn KNN",
    "weka": "Weka IBk",
    "custom_cpp": "Pure C++20 KNN (experimental)",
}
COLORS = {
    "custom_python_v5_1": "#4c78a8",
    "sklearn": "#f2a541",
    "weka": "#59a14f",
    "custom_cpp": "#e15759",
}
THREAD_ENVIRONMENT = SINGLE_THREAD_ENVIRONMENT


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def subprocess_environment():
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    return environment


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty series")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    standard_deviation = math.sqrt(variance)
    return {
        "count": len(values),
        "median_seconds": percentile(values, 0.5),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mean_seconds": mean,
        "sample_std_seconds": standard_deviation,
        "q1_seconds": q1,
        "q3_seconds": q3,
        "iqr_seconds": q3 - q1,
        "p10_seconds": percentile(values, 0.10),
        "p90_seconds": percentile(values, 0.90),
        "coefficient_of_variation": standard_deviation / mean if mean else None,
    }


def make_python_models(k):
    return {
        "custom_python_v5_1": CustomKNNClassifier(k, DEFAULT_BATCH_SIZE),
        "sklearn": KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            algorithm="brute",
            metric="euclidean",
            n_jobs=1,
        ),
    }


def timed_python_prediction(classifier, queries):
    start = perf_counter()
    probabilities = classifier.predict_proba(queries)
    labels = classifier.classes_[np.argmax(probabilities, axis=1)]
    elapsed = perf_counter() - start
    votes = np.rint(probabilities[:, 1] * classifier.n_neighbors).astype(np.int64)
    return {
        "seconds": elapsed,
        "prediction_hash": prediction_hash(labels, votes),
        "labels": labels,
        "positive_vote_counts": votes,
    }


def run_cpp_prediction(
    executable,
    cpp_input_dir,
    output,
    k,
    warmups=1,
    runs=1,
    batch_size=32,
    selection="heap",
    input_files=None,
):
    if input_files is None:
        input_files = {
            "train_features": cpp_input_dir / "train_features.csv",
            "train_labels": cpp_input_dir / "train_labels.csv",
            "test_features": cpp_input_dir / "test_features.csv",
            "test_labels": cpp_input_dir / "test_labels.csv",
        }
    command = [
        str(Path(executable).resolve()),
        "--train-features", str(Path(input_files["train_features"]).resolve()),
        "--train-labels", str(Path(input_files["train_labels"]).resolve()),
        "--test-features", str(Path(input_files["test_features"]).resolve()),
        "--test-labels", str(Path(input_files["test_labels"]).resolve()),
        "--k", str(k),
        "--output", str(Path(output).resolve()),
        "--warmups", str(warmups),
        "--runs", str(runs),
        "--batch-size", str(batch_size),
        "--selection", selection,
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=subprocess_environment(),
    )
    return read_json(output)


def run_weka_prediction(output, k, train=None, test=None):
    output = Path(output)
    train = Path(train) if train is not None else PROCESSED_DATA_DIR / "train.arff"
    test = Path(test) if test is not None else PROCESSED_DATA_DIR / "test.arff"
    command = [
        "java",
        "-XX:ActiveProcessorCount=1",
        "-jar", str((WEKA_DIR / "target/knn-weka-runner.jar").resolve()),
        "--train", str(train.resolve()),
        "--test", str(test.resolve()),
        "--k", str(k),
        "--predictions", str(output.resolve()),
        "--selected-parameters", str((RESULTS_DIR / "selected_parameters.json").resolve()),
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        env=subprocess_environment(),
    )
    runtime = read_json(output.with_suffix(".runtime.json"))
    runtime["prediction_file_sha256"] = sha256_file(output)
    return runtime


def command_version(command):
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            env=subprocess_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    combined = (completed.stdout + completed.stderr).strip()
    return combined.splitlines()[0] if combined else "unavailable"


def cmake_version():
    executable = shutil.which("cmake")
    if executable is None and sys.platform == "win32":
        candidate = Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE"
            r"\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        )
        if candidate.is_file():
            executable = str(candidate)
    return command_version([executable, "--version"]) if executable else "unavailable"


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except (ImportError, OSError):
            pass
    return platform.processor() or platform.machine()


def total_ram_bytes():
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
        except (AttributeError, OSError):
            pass
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return None


def weka_version():
    pom = (WEKA_DIR / "pom.xml").read_text(encoding="utf-8")
    match = re.search(
        r"<artifactId>weka-stable</artifactId>\s*<version>([^<]+)</version>", pom
    )
    return match.group(1) if match else "unknown"


def collect_environment(threadpools=None, cpp_metadata=None):
    if sys.platform == "win32":
        raw_power_mode = command_version(["powercfg", "/getactivescheme"])
        power_guid = re.search(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            raw_power_mode,
        )
        if power_guid:
            guid = power_guid.group(0).lower()
            known_name = {
                "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
                "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
                "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
            }.get(guid, "Power plan")
            power_mode = f"{known_name} ({guid})"
        else:
            power_mode = "unavailable"
    else:
        power_mode = "not queried"
    compact_pools = []
    for pool in threadpools or []:
        compact_pools.append({
            key: pool.get(key)
            for key in (
                "user_api", "internal_api", "prefix", "version",
                "num_threads", "threading_layer", "architecture",
            )
        })
    physical_cpu_count = None
    affinity = None
    try:
        import psutil

        physical_cpu_count = psutil.cpu_count(logical=False)
        affinity = psutil.Process().cpu_affinity() if hasattr(psutil.Process(), "cpu_affinity") else None
    except (ImportError, OSError, AttributeError):
        pass
    cpp_metadata = cpp_metadata or {}
    executable_path = cpp_metadata.get("executable")
    executable_sha256 = (
        sha256_file(executable_path)
        if executable_path and Path(executable_path).is_file()
        else cpp_metadata.get("executable_sha256", "not captured")
    )
    return {
        "captured_at": timestamp(),
        "cpu": cpu_name(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": physical_cpu_count,
        "process_affinity": affinity,
        "ram_bytes": total_ram_bytes(),
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
        "java": command_version(["java", "-version"]),
        "weka": weka_version(),
        "cmake": cmake_version(),
        "cpp_compiler": cpp_metadata.get("compiler", "not captured"),
        "cpp_build_mode": cpp_metadata.get("build_mode", "not captured"),
        "cpp_compile_flags": cpp_metadata.get(
            "compile_flags", "/O2 /arch:AVX2 /fp:precise; IPO/LTO enabled"
        ),
        "cpp_native_optimization": cpp_metadata.get("build_mode") == "Release native",
        "cpp_lto": cpp_metadata.get("build_mode") == "Release native",
        "cpp_executable_sha256": executable_sha256,
        "power_mode": power_mode,
        "thread_environment": THREAD_ENVIRONMENT,
        "numerical_threadpools": compact_pools,
        "machine_state_note": (
            "CPU frequency, thermal state, and background activity were not locked; "
            "raw order and outliers are retained."
        ),
    }
