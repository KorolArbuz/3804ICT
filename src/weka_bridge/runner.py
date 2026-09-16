"""Build, verify and run genuine Weka IBk on shared final ARFF matrices."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import shutil
import subprocess
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEKA_DIR = PROJECT_ROOT / "weka"
JAR = WEKA_DIR / "target/knn-weka-runner.jar"
BUILD_MANIFEST = WEKA_DIR / "target/build_manifest.json"
FINAL_THRESHOLD = 0.3315411365543412


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [WEKA_DIR / "pom.xml", *sorted((WEKA_DIR / "src").rglob("*.java"))]
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha(path) for path in paths}


def find_maven():
    """Use an existing installation; never download or install a toolchain."""
    executable = shutil.which("mvn")
    if executable:
        return executable
    for key in ("MAVEN_HOME", "M2_HOME"):
        if os.environ.get(key):
            for name in ("mvn.cmd", "mvn"):
                candidate = Path(os.environ[key]) / "bin" / name
                if candidate.is_file():
                    return str(candidate)
    # IntelliJ bundles Maven. This optional discovery has no username/install-version constant.
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates = sorted((Path(local) / "Programs").glob("IntelliJ*/plugins/maven/lib/maven3/bin/mvn.cmd"))
        if candidates:
            return str(candidates[-1])
    return None


def _run(command, *, timeout, cwd=PROJECT_ROOT):
    """Communicate with cleanup on timeout, interrupt or other exceptions."""
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace",
                          start_new_session=(os.name != "nt")) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            _kill_tree(process)
            process.communicate(timeout=10)
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _kill_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        finally:
            if process.poll() is None:
                process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)


def verify_artifact():
    if not JAR.is_file() or not BUILD_MANIFEST.is_file():
        raise FileNotFoundError("Fresh Weka JAR/build manifest missing; run with --build (requires Maven and JDK)")
    manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_hashes") != source_hashes() or manifest.get("artifact_sha256") != _sha(JAR):
        raise RuntimeError("Weka JAR is stale or modified; rebuild with --build")
    return manifest


def build(force=False, timeout=600):
    if not force:
        verify_artifact()
        return JAR
    maven = find_maven()
    if maven is None:
        raise FileNotFoundError("Maven unavailable: install Maven separately and add mvn to PATH or set MAVEN_HOME")
    before = source_hashes()
    cache = PROJECT_ROOT / ".maven-cache"
    command = [maven, "-B", "-ntp", f"-Dmaven.repo.local={cache}", "-f", str(WEKA_DIR / "pom.xml"), "clean", "package"]
    completed = _run(command, timeout=timeout)
    log = WEKA_DIR / "target/maven_build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode or not JAR.is_file():
        raise RuntimeError(f"Maven clean build/tests failed ({completed.returncode}); see {log}")
    if before != source_hashes():
        raise RuntimeError("Weka sources changed during build; rerun build")
    version = _run([maven, "-version"], timeout=30)
    manifest = {"source_hashes": before, "artifact_sha256": _sha(JAR),
                "weka_version": "3.8.6", "build_command": command,
                "toolchain": version.stdout + version.stderr, "java_release": 11,
                "tests": "Maven clean package including Surefire tests succeeded"}
    BUILD_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return JAR


def _command(train, test, k, threshold):
    java = shutil.which("java")
    if java is None:
        raise FileNotFoundError("Java unavailable; install a JDK 11 or later and add java to PATH")
    if k < 1 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid k or threshold")
    for path in (train, test):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return [java, "-XX:ActiveProcessorCount=1", "-jar", str(JAR), "--train", str(Path(train).resolve()),
            "--test", str(Path(test).resolve()), "--k", str(k), "--threshold", repr(threshold)]


def run_final(train, test, predictions, *, threshold=FINAL_THRESHOLD, k=101,
              force_rebuild=False, timeout=600):
    build(force_rebuild, timeout=timeout)
    predictions = Path(predictions).resolve()
    runtime_path = predictions.with_suffix(".runtime.json")
    if predictions.exists() or runtime_path.exists():
        raise FileExistsError("Refusing to overwrite existing Weka predictions/runtime artifacts")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    command = prediction_command(train, test, predictions, k=k, threshold=threshold)
    completed = _run(command, timeout=timeout)
    log = predictions.with_suffix(".execution.log")
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Java/Weka failed ({completed.returncode}); see {log}")
    if not predictions.is_file() or not runtime_path.is_file():
        raise RuntimeError("Java/Weka completed without prediction/runtime artifacts")
    response = json.loads(completed.stdout)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if response.get("status") != "COMPLETE" or response.get("checksum") != runtime.get("checksum"):
        raise RuntimeError("Malformed Weka completion protocol")
    runtime.update({"artifact_sha256": verify_artifact()["artifact_sha256"],
                    "source_hashes": source_hashes(), "process_mode": "fresh JVM; no warmup"})
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    return runtime


def prediction_command(train, test, predictions, *, k=101, threshold=FINAL_THRESHOLD):
    """Prepare checked argv outside a timer; executing it includes fresh JVM/load/fit/predict/output."""
    verify_artifact()
    return [*_command(train, test, k, threshold), "--predictions", str(Path(predictions).resolve())]


class WekaWorker:
    """Persistent sequential JVM; warmups and timed searches share one process."""

    def __init__(self, train, test, *, k=101, threshold=FINAL_THRESHOLD, warmups=3,
                 timeout=600, log_path=None):
        if warmups < 0:
            raise ValueError("Negative warmups")
        artifact = verify_artifact()
        command = [*_command(train, test, k, threshold), "--worker", "--warmups", str(warmups)]
        self.timeout = timeout
        self._queue = queue.Queue()
        self.process = None
        log_path = Path(log_path) if log_path else Path(test).parent / "weka_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self._log, text=True,
                                            encoding="utf-8", errors="replace", bufsize=1,
                                            start_new_session=(os.name != "nt"))
            self._reader = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader.start()
            self.ready = self._receive("READY")
            self.ready["artifact_sha256"] = artifact["artifact_sha256"]
            self.ready["process_mode"] = "persistent JVM; warmups and predictions in same process"
            checksums = {row["checksum"] for row in self.ready["warmups"]}
            if len(checksums) > 1:
                raise RuntimeError("Weka predictions changed during warmup")
            self._checksum = next(iter(checksums), None)
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self._queue.put(line)
        except (OSError, ValueError):
            pass  # Stream closed during cleanup after an exception/interrupt.
        finally:
            self._queue.put(None)

    def _receive(self, expected):
        try:
            line = self._queue.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Weka worker timed out waiting for {expected}") from exc
        if line is None:
            raise RuntimeError(f"Weka worker exited before {expected}; see worker log")
        message = json.loads(line)
        if message.get("status") != expected:
            raise RuntimeError(f"Malformed Weka worker response: {message}")
        return message

    def predict(self):
        try:
            self.process.stdin.write("PREDICT\n")
            self.process.stdin.flush()
            response = self._receive("PREDICT")
            if not math.isfinite(response["seconds"]) or response["seconds"] <= 0:
                raise RuntimeError("Invalid Weka timing")
            if response["n_query"] != self.ready["n_query"]:
                raise RuntimeError("Weka query count changed")
            if self._checksum is not None and response["checksum"] != self._checksum:
                raise RuntimeError("Weka predictions changed between calls")
            self._checksum = response["checksum"]
            return response
        except BaseException:
            self.close()
            raise

    def close(self):
        process = self.process
        try:
            if process and process.poll() is None:
                try:
                    process.stdin.write("EXIT\n")
                    process.stdin.flush()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    _kill_tree(process)
                    process.wait(timeout=10)
        finally:
            if process:
                if process.stdin:
                    process.stdin.close()
                if process.stdout:
                    process.stdout.close()
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run(train, test, k, predictions, selected_parameters=None, force_rebuild=False,
        metric="euclidean", weights="distance", threshold=FINAL_THRESHOLD):
    """Compatibility name for callers migrating to the final prepared-input adapter."""
    if metric != "euclidean" or weights != "distance" or selected_parameters is not None:
        raise ValueError("Active Weka adapter uses frozen final Euclidean/inverse configuration; historical runner is archived")
    return run_final(train, test, predictions, k=k, threshold=threshold, force_rebuild=force_rebuild)
