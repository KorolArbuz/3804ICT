"""Benchmark provenance and matched-trial ratios; never performs model inference."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_KEYS = [
    "run_id",
    "phase",
    "trial",
    "config_hash",
    "n_train",
    "n_query",
    "n_features",
    "k",
]


def process_affinity(process_id):
    """Observe an available OS CPU mask without changing affinity or priority."""
    try:
        if hasattr(os, "sched_getaffinity"):
            return {
                "cpus": sorted(os.sched_getaffinity(process_id)),
                "status": "observed",
            }
        if os.name == "nt":
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetProcessAffinityMask.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            kernel.GetProcessAffinityMask.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, process_id)
            if not handle:
                raise OSError(ctypes.get_last_error(), "OpenProcess failed")
            try:
                process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
                if not kernel.GetProcessAffinityMask(
                    handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
                ):
                    raise OSError(
                        ctypes.get_last_error(), "GetProcessAffinityMask failed"
                    )
                return {
                    "cpus": [
                        bit
                        for bit in range(ctypes.sizeof(process_mask) * 8)
                        if process_mask.value & (1 << bit)
                    ],
                    "status": "observed",
                    "scope": "process primary Windows processor group",
                }
            finally:
                kernel.CloseHandle(handle)
        return {"cpus": None, "status": "unavailable on this OS"}
    except (OSError, AttributeError) as error:
        return {"cpus": None, "status": "unavailable", "reason": str(error)}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def implementation_provenance(
    name, manifest, environment, ready, artifact_sha256, process_id
):
    prefixes = (
        ("src/custom_knn/", "src/final_knn/runner.py")
        if name == "baseline_python_v5_1"
        else ("src/cpp_knn/", "CMakeLists.txt")
        if name == "final_cpp"
        else ("weka/", "src/weka_bridge/")
        if name == "final_weka"
        else ("src/final_knn/", "src/preprocessing/")
    )
    sources = {
        key: value
        for key, value in manifest["sources"].items()
        if key.startswith(prefixes)
    }
    source_digest = hashlib.sha256(_json(sources).encode("utf-8")).hexdigest()
    native = name in ("final_cpp", "final_weka")
    return {
        "implementation_source_sha256": source_digest,
        "compiled_source_id": ready.get("source_id"),
        "artifact_sha256": artifact_sha256,
        "artifact_kind": "executable"
        if name == "final_cpp"
        else "weka_jar"
        if name == "final_weka"
        else "python_interpreter",
        "compiler_or_runtime": ready.get(
            "compiler", ready.get("java_version", environment.get("python"))
        ),
        "build_flags_or_options": ready.get(
            "flags",
            ready.get(
                "classifier_options",
                "n_jobs=1; brute; distance"
                if name == "final_sklearn"
                else "Python binary64",
            ),
        ),
        "library_versions": _json(
            {"weka": ready.get("weka_version")}
            if name == "final_weka"
            else {}
            if name == "final_cpp"
            else environment["packages"]
        ),
        "thread_environment": _json(environment["thread_environment"]),
        "threadpool_info": _json([] if native else environment["threadpools"]),
        "knn_compute_threads": 1,
        "thread_scope": "sequential KNN; JVM service/JIT/GC threads may exist"
        if name == "final_weka"
        else "one KNN compute thread; process thread total not claimed",
        "affinity": _json(process_affinity(process_id)),
        "affinity_scope": "resident process observed before warmups; no affinity changes requested",
        "temperature_celsius": None,
        "cpu_frequency_mhz": None,
    }


def paired_ratios(raw):
    """Only compare matching final workload/session/trial records, never baseline."""
    required = set(PAIR_KEYS + ["implementation_id", "seconds", "is_warmup"])
    if not required.issubset(raw.columns):
        raise ValueError(
            f"Missing matched-trial keys: {sorted(required - set(raw.columns))}"
        )
    warmup = raw.is_warmup.astype(str).str.lower().isin({"true", "1"})
    measured = raw[~warmup & raw.implementation_id.str.startswith("final_")].copy()
    if not np.isfinite(measured.seconds).all() or (measured.seconds <= 0).any():
        raise ValueError("Matched ratios require finite positive timings")
    if measured.duplicated(PAIR_KEYS + ["implementation_id"]).any():
        raise ValueError(
            "Duplicate implementation trial would create ambiguous pairing"
        )
    reference = measured[measured.implementation_id == "final_python"][
        PAIR_KEYS + ["seconds"]
    ].rename(columns={"seconds": "reference_seconds"})
    candidates = measured.rename(columns={"seconds": "implementation_seconds"})
    pairs = candidates.merge(
        reference, on=PAIR_KEYS, how="inner", validate="many_to_one"
    )
    pairs["reference_implementation"] = "final_python"
    pairs["final_python_over_implementation"] = (
        pairs.reference_seconds / pairs.implementation_seconds
    )
    return pairs[
        PAIR_KEYS
        + [
            "reference_implementation",
            "implementation_id",
            "reference_seconds",
            "implementation_seconds",
            "final_python_over_implementation",
        ]
    ]


def write_evidence(run_dir, raw, setup):
    """Emit derived tables while preserving all measured samples and values."""
    run_dir = Path(run_dir)
    pairs = paired_ratios(raw)
    pairs.to_csv(
        run_dir / "benchmark_paired_ratios.csv", index=False, float_format="%.17g"
    )
    rows = []
    for name, data in setup.items():
        ready = data["ready"]
        rows.append(
            {
                "implementation_id": name,
                "worker_setup_seconds": data["worker_setup_seconds"],
                "fit_seconds": data["fit_seconds"],
                "fit_scope": data["fit_scope"],
                "worker_setup_scope": data["worker_setup_scope"],
                "artifact_sha256": data["provenance"]["artifact_sha256"],
                "compiler_or_runtime": data["provenance"]["compiler_or_runtime"],
                "n_train": ready.get("training_rows", ready.get("n_train")),
                "n_query": ready.get("test_rows", ready.get("n_query")),
            }
        )
    pd.DataFrame(rows).to_csv(
        run_dir / "benchmark_setup.csv", index=False, float_format="%.17g"
    )
    return pairs
