"""Capture reproducibility metadata without rerunning timing measurements."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_info, threadpool_limits

from src.common.config import RESULTS_DIR

from .benchmark_utils import BENCHMARK_DIR, collect_environment, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BENCHMARK_DIR / "environment_comprehensive.json"
    )
    parser.add_argument(
        "--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe")
    )
    args = parser.parse_args(argv)
    output = args.output
    previous = read_json(output) if output.exists() else {}
    correctness = read_json(RESULTS_DIR / "cpp_knn_correctness.json")
    cpp_metadata = {
        "compiler": correctness["cpp"]["compiler"],
        "build_mode": correctness["cpp"]["build_mode"],
        "compile_flags": "/O2 /arch:AVX2 /fp:precise; IPO/LTO enabled",
        "executable": args.cpp_executable,
    }
    with threadpool_limits(limits=1):
        environment = collect_environment(threadpool_info(), cpp_metadata)
    protocol = {
        "selected_k": 19,
        "features": 33,
        "training_rows": 24000,
        "test_rows": 6000,
        "random_seed": 3804,
        "prediction_runs_each": 20,
        "prediction_warmups_each": 1,
        "full_pipeline_runs_each": 5,
        "scaling_runs_python_sklearn_cpp": 5,
        "scaling_runs_weka": 3,
        "training_sizes": [1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000],
        "fixed_training_scaling_query_rows": 1000,
        "query_sizes": [100, 250, 500, 1000, 2000, 4000, 6000],
        "cpp_configuration_runs_each": 5,
        "cpp_process_mode_runs_each": 20,
        "python_thread_policy": "environment variables plus threadpoolctl limit=1",
        "weka_policy": "existing IBk runner in a one-CPU JVM",
        "cpp_policy": "single-threaded exact native Release candidate",
    }
    protocol.update(previous.get("protocol", {}))
    environment["protocol"] = protocol
    environment["telemetry_limitation"] = (
        "Reliable per-run CPU effective frequency and temperature were unavailable "
        "without adding a machine-specific dependency, so those fields are blank."
    )
    write_json(output, environment)
    print(f"Environment metadata -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
