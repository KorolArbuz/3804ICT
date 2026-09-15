"""Screen supported pure C++20 build, selection, and batch configurations."""

import argparse
import random
import tempfile
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.cpp_knn.evidence import sha256_file

from .benchmark_utils import BENCHMARK_DIR, read_json, run_cpp_prediction, timestamp, write_rows


FIELDS = (
    "global_configuration_order",
    "build",
    "selection_method",
    "batch_size",
    "run_number",
    "runtime_seconds",
    "warmups",
    "compiler",
    "build_type",
    "native_optimization",
    "lto",
    "executable_sha256",
    "prediction_hash",
    "peak_auxiliary_bytes_estimate",
    "timestamp",
    "selected_main_configuration",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--native", type=Path, default=Path("build-cpp-native/cpp_knn.exe"))
    parser.add_argument("--portable", type=Path, default=Path("build-cpp-portable/cpp_knn.exe"))
    parser.add_argument("--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp")
    parser.add_argument("--seed", type=int, default=3804)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "raw_cpp_configuration.csv")
    args = parser.parse_args(argv)
    if args.runs < 3:
        raise ValueError("Use at least three measurements per C++ configuration")
    correctness = read_json(RESULTS_DIR / "cpp_knn_correctness.json")
    if not correctness.get("passed"):
        raise AssertionError("C++ correctness must pass before configuration timing")
    expected_hash = correctness["accepted_v5_1"]["prediction_hash"]
    candidates = [
        (build, executable, selection, batch)
        for build, executable in (("portable", args.portable), ("native", args.native))
        for selection, batch in (("heap", 16), ("heap", 32), ("heap", 64), ("nth", 32))
    ]
    for _, executable, _, _ in candidates:
        if not executable.is_file():
            raise FileNotFoundError(f"Missing C++ executable: {executable}")
    random.Random(args.seed).shuffle(candidates)
    rows = []
    with tempfile.TemporaryDirectory(prefix="cpp_configuration_screen_") as directory:
        temporary = Path(directory)
        for order, (build, executable, selection, batch_size) in enumerate(candidates, 1):
            result = run_cpp_prediction(
                executable,
                args.cpp_input_dir,
                temporary / f"{order}_{build}_{selection}_{batch_size}.json",
                19,
                warmups=1,
                runs=args.runs,
                batch_size=batch_size,
                selection=selection,
            )
            if result["prediction_hash"] != expected_hash:
                raise AssertionError(f"Incorrect output for {build}/{selection}/{batch_size}")
            executable_hash = sha256_file(executable)
            for run_number, seconds in enumerate(result["prediction_seconds"], 1):
                rows.append({
                    "global_configuration_order": order,
                    "build": build,
                    "selection_method": "nth_element" if selection == "nth" else "bounded max-heap",
                    "batch_size": batch_size,
                    "run_number": run_number,
                    "runtime_seconds": f"{float(seconds):.17g}",
                    "warmups": 1,
                    "compiler": result["compiler"],
                    "build_type": result["build_mode"],
                    "native_optimization": str(build == "native").lower(),
                    "lto": str(build == "native").lower(),
                    "executable_sha256": executable_hash,
                    "prediction_hash": result["prediction_hash"],
                    "peak_auxiliary_bytes_estimate": result["peak_auxiliary_bytes_estimate"],
                    "timestamp": timestamp(),
                    "selected_main_configuration": str(
                        build == "native" and selection == "heap" and batch_size == 32
                    ).lower(),
                })
            print(f"C++ screen: {build}/{selection}/batch {batch_size}", flush=True)
    write_rows(args.output, rows, FIELDS)
    print(f"C++ configuration screen: {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
