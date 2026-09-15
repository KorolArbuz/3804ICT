"""Optional CPU-affinity stability diagnostic for an explicitly selected CPU."""

import argparse
import os
import random
import tempfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair, read_json as read_project_json
from src.cpp_knn.evidence import sha256_file

from .benchmark_utils import (
    BENCHMARK_DIR,
    make_python_models,
    read_json,
    run_cpp_prediction,
    summarize,
    timed_python_prediction,
    timestamp,
    write_json,
    write_rows,
)


IMPLEMENTATIONS = ("custom_python_v5_1", "sklearn", "custom_cpp")
FIELDS = (
    "affinity_mode", "cpu_index", "run_number", "implementation",
    "runtime_seconds", "warmup", "prediction_hash", "timestamp", "notes",
)


class ProcessAffinity:
    def __init__(self):
        if os.name == "nt":
            import ctypes
            self._ctypes = ctypes
            self._kernel = ctypes.windll.kernel32
            self._kernel.GetCurrentProcess.restype = ctypes.c_void_p
            self._kernel.GetProcessAffinityMask.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            )
            self._kernel.SetProcessAffinityMask.argtypes = (
                ctypes.c_void_p,
                ctypes.c_size_t,
            )
            self._process = self._kernel.GetCurrentProcess()
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            if not self._kernel.GetProcessAffinityMask(
                self._process, ctypes.byref(process_mask), ctypes.byref(system_mask)
            ):
                raise OSError("GetProcessAffinityMask failed")
            self.original = {index for index in range(process_mask.value.bit_length()) if process_mask.value & (1 << index)}
        elif hasattr(os, "sched_getaffinity"):
            self.original = set(os.sched_getaffinity(0))
        else:
            raise RuntimeError("CPU affinity is not supported on this platform")

    def set(self, cpus):
        cpus = set(cpus)
        if not cpus or not cpus.issubset(self.original):
            raise ValueError(f"Requested CPUs must be within the current affinity: {sorted(self.original)}")
        if os.name == "nt":
            mask = sum(1 << cpu for cpu in cpus)
            if not self._kernel.SetProcessAffinityMask(
                self._process, self._ctypes.c_size_t(mask)
            ):
                raise OSError("SetProcessAffinityMask failed")
        else:
            os.sched_setaffinity(0, cpus)

    def restore(self):
        self.set(self.original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-index", type=int, required=True, help="Logical CPU chosen by the user; never inferred")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cpp-executable", type=Path, default=Path("build-cpp-native/cpp_knn.exe"))
    parser.add_argument("--cpp-input-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "affinity_stability_diagnostic.csv")
    parser.add_argument("--seed", type=int, default=3804)
    args = parser.parse_args(argv)
    if args.runs < 10:
        raise ValueError("At least 10 controlled runs per affinity mode are required")
    if not read_json(RESULTS_DIR / "cpp_knn_correctness.json").get("passed"):
        raise AssertionError("The C++ real-data verifier must pass before timing")

    affinity = ProcessAffinity()
    if args.cpu_index not in affinity.original:
        raise ValueError(f"CPU {args.cpu_index} is outside current affinity {sorted(affinity.original)}")
    training, testing = load_pair(PROCESSED_DATA_DIR / "train.npz", PROCESSED_DATA_DIR / "test.npz")
    k = int(read_project_json(RESULTS_DIR / "selected_parameters.json")["selected_k"])
    models = make_python_models(k)
    rows = []
    randomizer = random.Random(args.seed)
    executable_hash = sha256_file(args.cpp_executable)
    expected_hashes = {}

    with tempfile.TemporaryDirectory(prefix="knn_affinity_") as directory:
        temporary = Path(directory)
        with threadpool_limits(limits=1):
            for model in models.values():
                model.fit(training["X"], training["y"])
            try:
                modes = [("unpinned", affinity.original), ("pinned", {args.cpu_index})]
                randomizer.shuffle(modes)
                global_order = 0
                for mode, cpus in modes:
                    affinity.set(cpus)
                    for run in range(0, args.runs + 1):
                        names = list(IMPLEMENTATIONS)
                        randomizer.shuffle(names)
                        for name in names:
                            global_order += 1
                            if name in models:
                                result = timed_python_prediction(models[name], testing["X"])
                                runtime, digest = result["seconds"], result["prediction_hash"]
                            else:
                                result = run_cpp_prediction(
                                    args.cpp_executable, args.cpp_input_dir,
                                    temporary / f"cpp_{global_order}.json", k, warmups=1, runs=1,
                                )
                                runtime, digest = result["prediction_seconds"][0], result["prediction_hash"]
                            expected_hashes.setdefault(name, digest)
                            if digest != expected_hashes[name]:
                                raise AssertionError(f"{name} output changed during affinity diagnostic")
                            rows.append({
                                "affinity_mode": mode,
                                "cpu_index": args.cpu_index if mode == "pinned" else "",
                                "run_number": run,
                                "implementation": name,
                                "runtime_seconds": runtime,
                                "warmup": str(run == 0).lower(),
                                "prediction_hash": digest,
                                "timestamp": timestamp(),
                                "notes": (
                                    f"Current process affinity restored after benchmark; C++ executable {executable_hash}"
                                ),
                            })
            finally:
                affinity.restore()
    if expected_hashes["custom_cpp"] != expected_hashes["custom_python_v5_1"]:
        raise AssertionError("C++ no longer exactly matches accepted V5.1")
    write_rows(args.output, rows, FIELDS)
    write_json(args.output.with_suffix(".json"), rows)
    summary_rows = []
    for mode in ("unpinned", "pinned"):
        for name in IMPLEMENTATIONS:
            values = [
                float(row["runtime_seconds"])
                for row in rows
                if row["warmup"] == "false"
                and row["affinity_mode"] == mode
                and row["implementation"] == name
            ]
            stats = summarize(values)
            low = stats["q1_seconds"] - 1.5 * stats["iqr_seconds"]
            high = stats["q3_seconds"] + 1.5 * stats["iqr_seconds"]
            summary_rows.append({
                "affinity_mode": mode,
                "cpu_index": args.cpu_index if mode == "pinned" else "",
                "implementation": name,
                "runs": stats["count"],
                "median_seconds": stats["median_seconds"],
                "iqr_seconds": stats["iqr_seconds"],
                "max_seconds": stats["max_seconds"],
                "outlier_count_1_5_iqr": sum(value < low or value > high for value in values),
            })
    write_rows(
        BENCHMARK_DIR / "affinity_stability_summary.csv",
        summary_rows,
        tuple(summary_rows[0]),
    )
    print(f"CPU-affinity diagnostic collected on explicit CPU {args.cpu_index} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
