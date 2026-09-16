"""Five independently executed implementations with explicit run provenance."""
import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.common.config import PROJECT_ROOT
from src.common.experiment import (
    IMPLEMENTATIONS,
    environment,
    load_configs,
    model_group,
    source_hashes,
    utc_now,
)
from src.common.utils import write_json
from src.evaluation.final import evaluate, prediction_frame
from src.final_knn.runner import PythonModel
from src.preprocessing.prepared import prepare


def _raw_predictions(path, expected_rows):
    frame = pd.read_csv(path, float_precision="round_trip")
    if list(frame.columns) != [
        "test_position",
        "score_class_1",
        "y_pred",
    ] or not np.array_equal(frame.test_position, np.arange(expected_rows)):
        raise ValueError("External prediction schema/row ordering differs")
    return frame.score_class_1.to_numpy(), frame.y_pred.to_numpy()


def run(args):
    configs = load_configs()
    run_id = (
        utc_now().replace(":", "").replace("-", "").replace("+", "_")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run_dir = Path(args.output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "diagnostics"):
        (run_dir / name).mkdir()
    selected = args.only or list(IMPLEMENTATIONS)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started": utc_now(),
        "status": "running",
        "requested_implementations": selected,
        "required_implementations": list(IMPLEMENTATIONS),
        "config_hashes": {key: value["config_hash"] for key, value in configs.items()},
        "sources": source_hashes(),
        "benchmark_requested": args.benchmark,
        "implementations": {},
        "limitations": [
            "Previously inspected test; this is software reproduction, not independent ML validation.",
            "Weka and sklearn retain their native numerical semantics.",
        ],
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "resolved_configs.json", configs)
    print(f"Run directory: {run_dir}", flush=True)
    completed = []
    models = {}
    cpp_executable = None
    try:
        prepared, data_manifest = prepare(args.data, run_dir / "data", configs)
        write_json(run_dir / "environment.json", environment())
        for name in selected:
            print(f"Executing {name}", flush=True)
            group = model_group(name)
            data = prepared[group]
            config = configs[group]
            start = time.perf_counter()
            try:
                if name in IMPLEMENTATIONS[:3]:
                    model = PythonModel(name, data["X_train"], data["y_train"], config)
                    models[name] = model
                    timing, scores, labels = model.measured_predict(data["X_test"])
                    info = {
                        "fit_seconds": model.fit_seconds,
                        "prediction_seconds": timing["seconds"],
                        "prediction_checksum": timing["checksum"],
                        "threads": 1,
                    }
                elif name == "final_cpp":
                    from src.cpp_knn.bridge import build, score, validate_executable

                    cpp_executable = (
                        build(
                            args.cpp_build,
                            log_dir=PROJECT_ROOT / "verification/build_logs",
                        )
                        if args.build
                        else args.cpp_executable
                    )
                    identity = validate_executable(cpp_executable, args.cpp_build)
                    cpp_executable = Path(identity["executable"])
                    raw = score(
                        cpp_executable,
                        data["directory"],
                        run_dir / "native" / "cpp",
                        config,
                        build_type=args.cpp_build,
                    )
                    scores, labels = _raw_predictions(
                        raw["predictions_path"], len(data["X_test"])
                    )
                    info = {
                        key: value
                        for key, value in raw.items()
                        if key
                        not in (
                            "command",
                            "executable",
                            "predictions_path",
                            "runtime_path",
                            "scores_class_1",
                            "predicted_labels",
                            "positive_vote_counts",
                        )
                    }
                else:
                    from src.weka_bridge.runner import build, run_final

                    if args.build:
                        build(force=True)
                    output = run_dir / "native/weka/predictions.csv"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    info = run_final(
                        data["directory"] / "train.arff",
                        data["directory"] / "test.arff",
                        output,
                        threshold=config["model"]["threshold"],
                        k=config["model"]["k"],
                    )
                    scores, labels = _raw_predictions(output, len(data["X_test"]))
                frame = prediction_frame(run_id, name, config, data, scores, labels)
                frame.to_csv(
                    run_dir / "predictions" / f"{name}.csv",
                    index=False,
                    float_format="%.17g",
                )
                manifest["implementations"][name] = {
                    "status": "PASS",
                    "end_to_end_adapter_seconds": time.perf_counter() - start,
                    "metadata": info,
                }
                completed.append(name)
            except Exception as error:
                unavailable = isinstance(error, FileNotFoundError) or any(
                    marker in str(error).lower()
                    for marker in (
                        "unavailable",
                        "is missing",
                        "run with --build",
                        "rebuild current sources",
                    )
                )
                failure_status = "BLOCKED" if unavailable else "FAIL"
                manifest["implementations"][name] = {
                    "status": failure_status,
                    "reason": str(error),
                    "exception_type": type(error).__name__,
                }
                print(f"{name}: {failure_status}: {error}", file=sys.stderr, flush=True)
            write_json(run_dir / "run_manifest.json", manifest)
        if not completed:
            raise RuntimeError("No implementation completed")
        table, agreement, frames = evaluate(
            run_dir, run_id, completed, configs, data_manifest
        )
        custom_pair = agreement[
            (agreement.implementation_a == "final_python")
            & (agreement.implementation_b == "final_cpp")
        ]
        if (
            len(custom_pair)
            and (
                (custom_pair.scores_over_tolerance > 0)
                | (custom_pair.label_disagreements > 0)
            ).any()
        ):
            raise ValueError(
                "Unexpected manual Python/C++ difference: investigate diagnostics before accepting this run"
            )
        if args.benchmark:
            from src.benchmarking.suite import benchmark

            benchmark(
                run_dir,
                completed,
                prepared,
                configs,
                models,
                cpp_executable,
                args.cpp_build,
                args.runs,
                args.warmups,
            )
        else:
            pd.DataFrame(
                columns=[
                    "implementation_id",
                    "phase",
                    "seconds",
                    "trial",
                    "sequence",
                    "order_in_trial",
                    "is_warmup",
                ]
            ).to_csv(run_dir / "benchmark_raw.csv", index=False)
            pd.DataFrame(
                columns=[
                    "implementation_id",
                    "phase",
                    "median_seconds",
                    "iqr_seconds",
                    "q25_seconds",
                    "q75_seconds",
                ]
            ).to_csv(run_dir / "benchmark_summary.csv", index=False)
        differences = bool(
            len(agreement)
            and (
                (agreement.scores_over_tolerance > 0)
                | (agreement.label_disagreements > 0)
            ).any()
        )
        manifest["status"] = (
            ("complete_with_documented_differences" if differences else "complete")
            if set(completed) == set(IMPLEMENTATIONS)
            else "partial"
        )
        manifest["completed_implementations"] = completed
        manifest["finished"] = utc_now()
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(
            run_dir / "validation.json",
            {
                "status": "integration_pass" if len(completed) == 5 else "partial",
                "scope": "current run prediction contract, metrics and pairwise all-row comparison; full build/clean verification is separate",
                "checks": [
                    {
                        "check_id": "prediction_contract",
                        "scope": "integration",
                        "status": "PASS",
                        "implementations": completed,
                    }
                ],
                "full_validation": False,
            },
        )
        from src.reporting.plots import generate_report

        generate_report(run_dir)
        print(table.to_string(index=False), flush=True)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["error"] = "Run interrupted by user or coordinator"
        manifest["finished"] = utc_now()
        write_json(run_dir / "run_manifest.json", manifest)
        return 130, run_dir
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["finished"] = utc_now()
        write_json(run_dir / "run_manifest.json", manifest)
        print(f"Run failed: {error}", file=sys.stderr, flush=True)
        return 1, run_dir
    return (0 if manifest["status"].startswith("complete") else 1), run_dir


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path)
    result.add_argument("--build", action="store_true")
    result.add_argument(
        "--cpp-build", choices=["portable", "native"], default="portable"
    )
    result.add_argument("--cpp-executable", type=Path)
    result.add_argument("--only", nargs="+", choices=IMPLEMENTATIONS)
    result.add_argument("--benchmark", action="store_true")
    result.add_argument("--runs", type=int, default=20)
    result.add_argument("--warmups", type=int, default=3)
    result.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "results/final"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("runs must be positive and warmups nonnegative")
    if args.only and len(set(args.only)) != len(args.only):
        raise SystemExit("--only contains duplicate implementations")
    with threadpool_limits(1):
        return run(args)[0]
