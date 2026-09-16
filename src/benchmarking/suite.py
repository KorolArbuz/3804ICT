"""Sequential, randomized, warmed resident prediction and fresh-process scopes."""
from contextlib import ExitStack
import os
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT
from src.common.experiment import model_group, sha256, utc_now
from src.common.utils import write_json
from .evidence import implementation_provenance, paired_ratios, write_evidence


def summarize(raw):
    rows = []
    measured = raw[~raw.is_warmup.astype(bool)]
    for (name, phase), group in measured.groupby(
        ["implementation_id", "phase"], sort=False
    ):
        values = group.seconds.to_numpy(dtype=float)
        q10, q25, q50, q75, q90 = np.percentile(
            values, [10, 25, 50, 75, 90], method="linear"
        )
        rows.append(
            {
                "implementation_id": name,
                "model_group": model_group(name),
                "phase": phase,
                "n": len(values),
                "median_seconds": q50,
                "mean_seconds": values.mean(),
                "min_seconds": values.min(),
                "max_seconds": values.max(),
                "std_seconds": values.std(ddof=1) if len(values) > 1 else 0.0,
                "q25_seconds": q25,
                "q75_seconds": q75,
                "iqr_seconds": q75 - q25,
                "p10_seconds": q10,
                "p90_seconds": q90,
            }
        )
    result = pd.DataFrame(rows)
    result["final_python_over_implementation"] = np.nan
    pairs = paired_ratios(raw)
    for (name, phase), group in pairs.groupby(
        ["implementation_id", "phase"], sort=False
    ):
        mask = (result.implementation_id == name) & (result.phase == phase)
        result.loc[
            mask, "final_python_over_implementation"
        ] = group.final_python_over_implementation.median()
        result.loc[mask, "paired_trial_count"] = len(group)
    result[
        "ratio_scope"
    ] = "median of paired final_python/implementation ratios; same run, phase, trial and workload keys"
    return result


def benchmark(
    run_dir,
    implementations,
    prepared,
    configs,
    models,
    cpp_executable,
    cpp_build,
    runs=20,
    warmups=3,
):
    from src.cpp_knn.bridge import PersistentCpp, command_arguments, validate_executable
    from src.weka_bridge.runner import WekaWorker

    raw = []
    rng = np.random.default_rng(20260916)
    workers = {}
    checksums = {}
    setup = {}
    sequence = 0
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    python_artifact = sha256(sys.executable)
    run_id = run_dir.name

    def record(name, phase, trial, order, warmup, seconds, checksum, process_id):
        nonlocal sequence
        data = prepared[model_group(name)]
        config = configs[model_group(name)]
        mode = (
            "fresh process startup/load/fit/predict/output"
            if phase == "prepared_pipeline"
            else "persistent JVM"
            if name == "final_weka"
            else "persistent C++ process"
            if name == "final_cpp"
            else "resident Python"
        )
        row = {
            "run_id": run_id,
            "implementation_id": name,
            "model_group": model_group(name),
            "config_hash": config["config_hash"],
            "phase": phase,
            "trial": trial,
            "sequence": sequence,
            "order_in_trial": order,
            "is_warmup": warmup,
            "timestamp": utc_now(),
            "seconds": float(seconds),
            "checksum": checksum,
            "n_train": len(data["X_train"]),
            "n_query": len(data["X_test"]),
            "n_features": data["X_train"].shape[1],
            "k": config["model"]["k"],
            "batch_size": 64
            if name == "baseline_python_v5_1"
            else 32
            if name == "final_cpp"
            else 1
            if name == "final_python"
            else None,
            "process_id": process_id,
            "process_mode": mode,
            "build_mode": cpp_build if name == "final_cpp" else "not_applicable",
            **setup[name]["provenance"],
        }
        if phase == "prepared_pipeline":
            row[
                "affinity_scope"
            ] = "resident mask observed before benchmark; fresh child inherits parent affinity, not independently sampled"
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid measured duration")
        raw.append(row)
        sequence += 1
        pd.DataFrame(raw).to_csv(
            run_dir / "benchmark_raw.csv", index=False, float_format="%.17g"
        )

    with ExitStack() as stack:
        for name in implementations:
            start = time.perf_counter()
            if name == "final_cpp":
                workers[name] = stack.enter_context(
                    PersistentCpp(
                        cpp_executable,
                        prepared["final"]["directory"],
                        configs["final"],
                        warmups=0,
                        build_type=cpp_build,
                    )
                )
            elif name == "final_weka":
                directory = prepared["final"]["directory"]
                workers[name] = stack.enter_context(
                    WekaWorker(
                        directory / "train.arff",
                        directory / "test.arff",
                        k=101,
                        threshold=configs["final"]["model"]["threshold"],
                        warmups=0,
                    )
                )
            ready = (
                workers[name].ready
                if name in workers
                else {"process": "resident Python", "pid": os.getpid()}
            )
            worker_setup_seconds = time.perf_counter() - start
            artifact = (
                workers[name].identity["executable_sha256"]
                if name == "final_cpp"
                else ready["artifact_sha256"]
                if name == "final_weka"
                else python_artifact
            )
            pid = workers[name].process.pid if name in workers else os.getpid()
            setup[name] = {
                "worker_setup_seconds": worker_setup_seconds,
                "fit_seconds": models[name].fit_seconds
                if name in models
                else ready.get("fit_seconds"),
                "fit_scope": "classifier constructor and fit on prepared training data; loading excluded",
                "worker_setup_scope": "Python model already resident; fit measured previously"
                if name in models
                else "process startup, prepared input loading, fit and READY handshake; no prediction warmups",
                "ready": ready,
                "provenance": implementation_provenance(
                    name, manifest, environment, ready, artifact, pid
                ),
            }
        write_json(run_dir / "benchmark_setup.json", setup)
        for trial in range(warmups + runs):
            order = rng.permutation(implementations).tolist()
            for position, name in enumerate(order):
                if name in models:
                    measured, _, _ = models[name].measured_predict(
                        prepared[model_group(name)]["X_test"]
                    )
                    pid = os.getpid()
                else:
                    measured = workers[name].predict()
                    pid = workers[name].process.pid
                checksum = measured["checksum"]
                if name in checksums and checksum != checksums[name]:
                    raise ValueError(
                        f"Prediction checksum changed between trials for {name}"
                    )
                checksums[name] = checksum
                record(
                    name,
                    "prediction",
                    trial - warmups if trial >= warmups else trial,
                    position,
                    trial < warmups,
                    measured["seconds"],
                    checksum,
                    pid,
                )
            print(f"Benchmark resident round {trial+1}/{runs+warmups}", flush=True)
    # New process for every implementation/trial. No compiler or preprocessing is timed.
    process_dir = run_dir / "native/prepared_pipeline"
    process_dir.mkdir(parents=True, exist_ok=True)
    for trial in range(5):
        for order, name in enumerate(rng.permutation(implementations).tolist()):
            directory = prepared[model_group(name)]["directory"]
            output = process_dir / f"{name}_{trial}.csv"
            if name in models:
                command = [
                    sys.executable,
                    "-m",
                    "src.benchmarking.prepared_child",
                    "--implementation",
                    name,
                    "--data-dir",
                    str(directory),
                    "--output",
                    str(output),
                ]
            elif name == "final_cpp":
                command = command_arguments(
                    cpp_executable, directory, configs["final"]
                ) + [
                    "--predictions",
                    str(output),
                    "--output",
                    str(output.with_suffix(".json")),
                ]
            else:
                from src.weka_bridge.runner import prediction_command

                command = prediction_command(
                    directory / "train.arff",
                    directory / "test.arff",
                    output,
                    k=101,
                    threshold=configs["final"]["model"]["threshold"],
                )
            start = time.perf_counter()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=600)
            except BaseException:
                process.kill()
                process.communicate()
                raise
            elapsed = time.perf_counter() - start
            if process.returncode:
                raise RuntimeError(f"Prepared pipeline {name} failed: {stderr}")
            from src.evaluation.final import read_predictions

            expected = read_predictions(run_dir / "predictions" / f"{name}.csv")
            actual = pd.read_csv(output, float_precision="round_trip")
            if (
                list(actual.columns) != ["test_position", "score_class_1", "y_pred"]
                or not np.array_equal(actual.test_position, np.arange(len(expected)))
                or len(actual) != len(expected)
                or not np.array_equal(actual.y_pred, expected.y_pred)
                or not np.array_equal(actual.score_class_1, expected.score_class_1)
            ):
                raise ValueError("Fresh process predictions changed")
            checksum = (
                "sha256:"
                + hashlib.sha256(
                    np.ascontiguousarray(actual.score_class_1, dtype="<f8").tobytes()
                    + np.ascontiguousarray(actual.y_pred, dtype="<i8").tobytes()
                ).hexdigest()
            )
            record(
                name,
                "prepared_pipeline",
                trial,
                order,
                False,
                elapsed,
                checksum,
                process.pid,
            )
        print(f"Benchmark fresh-process round {trial+1}/5", flush=True)
    summary = summarize(pd.DataFrame(raw))
    summary.to_csv(run_dir / "benchmark_summary.csv", index=False, float_format="%.17g")
    write_evidence(run_dir, pd.DataFrame(raw), setup)
    write_json(
        run_dir / "benchmark_protocol.json",
        {
            "prediction": "ready resident model/data; scores and labels freshly computed/materialized/validated; output allocation included; timer ends before checksum and IPC serialization",
            "warmups": warmups,
            "measured_trials": runs,
            "prepared_pipeline_trials": 5,
            "prepared_pipeline": "fresh process startup + canonical prepared file loading + fit + scores/labels + CSV output; Python/C++ CSV, Weka ARFF; excludes raw data preprocessing/builds",
            "execution": "sequential randomized implementation order in each trial, seed 20260916",
            "percentiles": "numpy.percentile method=linear",
            "outliers": "none discarded",
            "groups": "baseline is a different representation/k/vote workload; ratios compare final implementations only",
            "thread_limits": 1,
            "artifact_hashes": {
                "cpp": sha256(cpp_executable) if cpp_executable else None
            },
            "environment_file": "environment.json",
        },
    )
