"""Compare fresh-process and persistent-process C++ prediction timings."""

import argparse
import random
import tempfile
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import read_json as read_project_json

from .benchmark_utils import (
    BENCHMARK_DIR,
    read_json,
    run_cpp_prediction,
    summarize,
    write_rows,
)


FIELDS = (
    "mode",
    "run",
    "runtime_seconds",
)


def _outlier_count(values):
    stats = summarize(values)
    low = stats["q1_seconds"] - 1.5 * stats["iqr_seconds"]
    high = stats["q3_seconds"] + 1.5 * stats["iqr_seconds"]
    return sum(value < low or value > high for value in values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe"))
    parser.add_argument("--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "cpp_process_mode_diagnostic.csv")
    parser.add_argument("--seed", type=int, default=3804)
    args = parser.parse_args(argv)
    if args.runs < 20:
        raise ValueError("At least 20 runs per C++ process mode are required")
    if not read_json(RESULTS_DIR / "cpp_knn_correctness.json").get("passed"):
        raise AssertionError("The C++ real-data verifier must pass before timing")
    k = int(read_project_json(RESULTS_DIR / "selected_parameters.json")["selected_k"])
    rows = []
    expected_hash = None
    block_order = ["fresh_process", "persistent_process"]
    random.Random(args.seed).shuffle(block_order)

    with tempfile.TemporaryDirectory(prefix="cpp_process_modes_") as directory:
        temporary = Path(directory)
        for mode in block_order:
            if mode == "fresh_process":
                for run in range(1, args.runs + 1):
                    result = run_cpp_prediction(
                        args.cpp_executable, args.cpp_input_dir,
                        temporary / f"fresh_{run}.json", k, warmups=1, runs=1,
                    )
                    digest = result["prediction_hash"]
                    expected_hash = expected_hash or digest
                    if digest != expected_hash:
                        raise AssertionError("C++ output changed in the fresh-process diagnostic")
                    rows.append({
                        "mode": mode,
                        "run": run,
                        "runtime_seconds": result["prediction_seconds"][0],
                    })
            else:
                result = run_cpp_prediction(
                    args.cpp_executable, args.cpp_input_dir,
                    temporary / "persistent.json", k, warmups=3, runs=args.runs,
                )
                digest = result["prediction_hash"]
                expected_hash = expected_hash or digest
                if digest != expected_hash:
                    raise AssertionError("C++ output changed in the persistent-process diagnostic")
                for run, runtime in enumerate(result["prediction_seconds"], start=1):
                    rows.append({
                        "mode": mode,
                        "run": run,
                        "runtime_seconds": runtime,
                    })

    write_rows(args.output, rows, FIELDS)
    summaries = []
    for mode in ("fresh_process", "persistent_process"):
        values = [float(row["runtime_seconds"]) for row in rows if row["mode"] == mode]
        stats = summarize(values)
        summaries.append({
            "mode": mode,
            "runs": stats["count"],
            "median_seconds": stats["median_seconds"],
            "iqr_seconds": stats["iqr_seconds"],
            "min_seconds": stats["min_seconds"],
            "max_seconds": stats["max_seconds"],
            "outlier_count_1_5_iqr": _outlier_count(values),
        })
    write_rows(BENCHMARK_DIR / "cpp_process_mode_summary.csv", summaries, tuple(summaries[0]))
    print("C++ process-mode diagnostic: " + ", ".join(
        f"{row['mode']}={row['median_seconds']:.4f}s median" for row in summaries
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
