"""Build and run the local C++20 implementation without installing a toolchain."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "CMakeLists.txt",
    "src/cpp_knn/knn.hpp",
    "src/cpp_knn/knn.cpp",
    "src/cpp_knn/data_io.hpp",
    "src/cpp_knn/data_io.cpp",
    "src/cpp_knn/main.cpp",
)


def source_id(root=ROOT):
    text = "".join(
        f"{name}:{hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()}\n"
        for name in SOURCE_FILES
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_cmake():
    """Discover PATH/CMAKE_COMMAND or an installed Visual Studio CMake."""
    override = os.environ.get("CMAKE_COMMAND")
    if override and Path(override).is_file():
        return Path(override)
    found = shutil.which("cmake")
    if found:
        return Path(found)
    if os.name == "nt":
        candidates = []
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            folder = Path(os.environ.get(variable, "C:/Program Files"))
            candidates.extend(
                folder.glob(
                    "Microsoft Visual Studio/*/*/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
                )
            )
        if candidates:
            return sorted(candidates)[-1]
    raise RuntimeError(
        "CMake is unavailable. Install CMake and a C++20 compiler, then rerun with --build."
    )


def executable_path(build_type="portable", build_dir=None):
    directory = (
        Path(build_dir) if build_dir else ROOT / "build" / f"final-cpp-{build_type}"
    )
    filename = "cpp_knn.exe" if os.name == "nt" else "cpp_knn"
    for candidate in (directory / "Release" / filename, directory / filename):
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"C++ executable is missing in {directory}; run with --build.")


def validate_executable(path=None, build_type="portable"):
    executable = Path(path).resolve() if path else executable_path(build_type)
    if not executable.is_file():
        raise RuntimeError(
            f"C++ executable is missing: {executable}; run with --build."
        )
    result = subprocess.run(
        [str(executable), "--identity"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    identity = json.loads(result.stdout)
    if identity.get("source_id") != source_id():
        raise RuntimeError(
            "C++ executable source identity is stale; rebuild current sources with --build."
        )
    if identity.get("build_mode") != build_type:
        raise RuntimeError("C++ executable build mode differs from --cpp-build.")
    if identity.get("build_configuration") != "Release":
        raise RuntimeError("C++ executable must be a Release build.")
    if identity.get("thread_count") != 1 or identity.get("cpp_standard") != 20:
        raise RuntimeError("C++ executable has an incompatible execution contract.")
    return {
        **identity,
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }


def build(build_type="portable", log_dir=None, *, sanitize=False, build_dir=None):
    if build_type not in ("portable", "native"):
        raise ValueError("build_type must be portable or native")
    cmake = find_cmake()
    suffix = "-sanitized" if sanitize else ""
    directory = (
        Path(build_dir)
        if build_dir
        else ROOT / "build" / f"final-cpp-{build_type}{suffix}"
    )
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logs = Path(log_dir) if log_dir else directory
    logs.mkdir(parents=True, exist_ok=True)
    configure = [
        str(cmake),
        "-S",
        str(ROOT),
        "-B",
        str(directory),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=ON",
        f"-DCPP_KNN_NATIVE={'ON' if build_type == 'native' else 'OFF'}",
        f"-DCPP_KNN_SANITIZE={'ON' if sanitize else 'OFF'}",
    ]
    if (
        os.name == "nt"
        and not (directory / "CMakeCache.txt").exists()
        and "Microsoft Visual Studio" in str(cmake)
    ):
        configure += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    ctest = cmake.with_name("ctest.exe" if os.name == "nt" else "ctest")
    commands = [
        configure,
        [str(cmake), "--build", str(directory), "--config", "Release"],
        [
            str(ctest),
            "--test-dir",
            str(directory),
            "-C",
            "Release",
            "--output-on-failure",
        ],
    ]
    records = []
    for name, command in zip(("configure", "build", "ctest"), commands):
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=600
        )
        log = logs / f"cpp_{build_type}{suffix}_{name}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        records.append(
            {"command": command, "exit_code": result.returncode, "log": str(log)}
        )
        (logs / f"cpp_{build_type}{suffix}_build.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(f"C++ {name} failed ({result.returncode}); see {log}.")
    executable = executable_path(build_type, directory)
    validate_executable(executable, build_type)
    return executable


def command_arguments(executable, data_dir, config, *, warmups=0):
    data_dir = Path(data_dir).resolve()
    model = config.get("model", config)
    return [
        str(Path(executable).resolve()),
        "--train-features",
        str(data_dir / "train_features.csv"),
        "--train-labels",
        str(data_dir / "train_labels.csv"),
        "--test-features",
        str(data_dir / "test_features.csv"),
        "--k",
        str(model["k"]),
        "--weights",
        str(model["weights"]),
        "--threshold",
        repr(float(model["threshold"])),
        "--model-id",
        "final_cpp",
        "--config-id",
        str(config["config_hash"]),
        "--batch-size",
        "32",
        "--warmups",
        str(warmups),
        "--runs",
        "1",
    ]


def score(
    executable,
    data_dir,
    output_dir,
    config,
    *,
    build_type="portable",
    algorithm="optimized",
):
    identity = validate_executable(executable, build_type)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / "predictions_cpp.csv"
    runtime = output_dir / "runtime_cpp.json"
    command = command_arguments(executable, data_dir, config) + [
        "--predictions",
        str(predictions),
        "--output",
        str(runtime),
        "--algorithm",
        algorithm,
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
    )
    (output_dir / "cpp.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    metadata = json.loads(runtime.read_text(encoding="utf-8"))
    if (
        metadata.get("config_id") != config["config_hash"]
        or metadata.get("source_id") != identity["source_id"]
    ):
        raise RuntimeError(
            "C++ output identity does not match the requested model/source."
        )
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata["test_rows"]:
        raise RuntimeError("C++ prediction count differs from query count.")
    for position, row in enumerate(rows):
        probability = float(row["score_class_1"])
        if (
            int(row["test_position"]) != position
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise RuntimeError("Malformed C++ prediction output.")
        if int(row["y_pred"]) != int(
            probability >= config.get("model", config)["threshold"]
        ):
            raise RuntimeError(
                "C++ final label does not implement the frozen threshold."
            )
    return {
        **metadata,
        **identity,
        "predictions_path": str(predictions),
        "runtime_path": str(runtime),
        "command": command,
    }


class PersistentCpp:
    """One warmed native process; commands execute new searches, with bounded IPC waits."""

    def __init__(
        self,
        executable,
        data_dir,
        config,
        warmups=3,
        *,
        build_type="portable",
        timeout=600,
    ):
        self.identity = validate_executable(executable, build_type)
        self.timeout = timeout
        self._stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self.process = subprocess.Popen(
            command_arguments(executable, data_dir, config, warmups=warmups)
            + ["--persistent", "true"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_lines, daemon=True)
        self._reader.start()
        try:
            self.ready = self._read("READY")
            if self.ready.get("config_id") != config["config_hash"]:
                raise RuntimeError("C++ persistent configuration mismatch.")
        except BaseException:
            self.close()
            raise

    def _read_lines(self):
        try:
            for line in self.process.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass  # Stream can close during exceptional process cleanup.
        finally:
            self._lines.put(None)

    def _read(self, status):
        try:
            line = self._lines.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("C++ worker response timed out") from error
        if line is None:
            self._stderr.seek(0)
            raise RuntimeError("C++ worker closed: " + self._stderr.read())
        value = json.loads(line)
        if value.get("status") != status:
            raise RuntimeError(f"Unexpected C++ worker response: {value}")
        return value

    def predict(self):
        try:
            self.process.stdin.write("PREDICT\n")
            self.process.stdin.flush()
            result = self._read("PREDICTION")
            if not math.isfinite(result["seconds"]) or result["seconds"] < 0:
                raise RuntimeError("Invalid native prediction duration")
            if result.get("prediction_count") != self.ready.get("test_rows"):
                raise RuntimeError("C++ query count changed")
            if not isinstance(result.get("checksum"), str) or not result[
                "checksum"
            ].startswith("fnv1a64:"):
                raise RuntimeError("Malformed C++ prediction checksum")
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe:
                pipe.close()
        self._stderr.close()
        self._reader.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
