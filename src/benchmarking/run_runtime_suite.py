"""Run all expensive measurements for the comprehensive runtime suite.

Chart and report generation are intentionally separate so they can be repeated
without rerunning benchmarks.
"""

import argparse
import tempfile
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR
from src.cpp_knn.verify import main as verify_cpp

from .benchmark_cpp_configurations import main as benchmark_cpp
from .benchmark_full_pipeline import main as benchmark_full_pipeline
from .benchmark_prediction_runs import main as benchmark_prediction
from .benchmark_scaling import main as benchmark_scaling
from .benchmark_utils import BENCHMARK_DIR, run_cpp_prediction
from .capture_environment import main as capture_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-runs", type=int, default=20)
    parser.add_argument("--full-pipeline-runs", type=int, default=5)
    parser.add_argument("--scaling-runs", type=int, default=5)
    parser.add_argument("--weka-scaling-runs", type=int, default=3)
    parser.add_argument("--cpp-configuration-runs", type=int, default=5)
    parser.add_argument("--cpp-native", default="build-cpp-native/cpp_knn.exe")
    parser.add_argument("--cpp-portable", default="build-cpp-portable/cpp_knn.exe")
    args = parser.parse_args(argv)
    native = Path(args.cpp_native)
    portable = Path(args.cpp_portable)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cpp_prebenchmark_verification_") as directory:
        cpp_result = Path(directory) / "cpp_result.json"
        run_cpp_prediction(
            native,
            PROCESSED_DATA_DIR / "cpp",
            cpp_result,
            19,
            warmups=1,
            runs=2,
        )
        verify_cpp([
            "--cpp-result", str(cpp_result),
            "--output", str(BENCHMARK_DIR / "cpp_correctness_comprehensive.json"),
        ])

    benchmark_prediction([
        "--runs", str(args.prediction_runs),
        "--cpp-executable", str(native),
    ])
    benchmark_full_pipeline([
        "--runs", str(args.full_pipeline_runs),
        "--cpp-executable", str(native),
        "--output", str(BENCHMARK_DIR / "raw_full_pipeline_runs.csv"),
    ])
    benchmark_scaling([
        "--runs", str(args.scaling_runs),
        "--weka-runs", str(args.weka_scaling_runs),
        "--cpp-executable", str(native),
    ])
    benchmark_cpp([
        "--runs", str(args.cpp_configuration_runs),
        "--native", str(native),
        "--portable", str(portable),
    ])
    capture_environment([
        "--cpp-executable", str(native),
    ])
    print("All comprehensive measurement stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
