"""Collect one named, non-overwriting controlled benchmark session."""

import argparse
import re
from pathlib import Path

from .benchmark_prediction_runs import main as benchmark_prediction
from .benchmark_utils import BENCHMARK_DIR, read_json, write_json
from .capture_environment import main as capture_environment


SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe"))
    args = parser.parse_args(argv)
    if not SESSION_ID.fullmatch(args.session_id):
        raise ValueError("Session id must contain only letters, digits, '.', '_' or '-'")

    session_dir = BENCHMARK_DIR / "sessions" / args.session_id
    if session_dir.exists():
        parser.error(f"session already exists and will not be overwritten: {session_dir}")
    session_dir.mkdir(parents=True)

    raw = session_dir / "raw_prediction_runs.csv"
    environment_path = session_dir / "environment.json"
    benchmark_prediction([
        "--runs", str(args.runs),
        "--cpp-executable", str(args.cpp_executable),
        "--output", str(raw),
    ])
    capture_environment([
        "--cpp-executable", str(args.cpp_executable),
        "--output", str(environment_path),
    ])
    environment = read_json(environment_path)
    environment["session_id"] = args.session_id
    environment["protocol"]["prediction_runs_each"] = args.runs
    environment["protocol"]["prediction_warmups_each"] = 1
    environment["protocol"]["session_is_independent"] = True
    write_json(environment_path, environment)
    print(f"Completed non-overwriting benchmark session -> {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
