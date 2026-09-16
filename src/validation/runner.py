"""Bounded release checks with explicit evidence, provenance and completion scopes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("build correctness", "same-semantics numerical correctness", "toolkit comparability",
          "real-data reproduction", "runtime evidence", "packaging completeness")
SCIENTIFIC_PREFIXES = ("src/custom_knn/", "src/final_knn/", "src/preprocessing/", "src/data/",
                       "src/common/", "src/cpp_knn/", "src/weka_bridge/", "weka/src/main/", "configs/")
SCIENTIFIC_FILES = {"run_all.py", "src/experiment.py", "CMakeLists.txt", "requirements.txt", "weka/pom.xml",
                    "src/benchmarking/suite.py", "src/benchmarking/prepared_child.py", "src/benchmarking/evidence.py"}
ARCHIVED_CPP_HELPERS = {"src/cpp_knn/benchmark.py", "src/cpp_knn/evidence.py",
                        "src/cpp_knn/export_data.py", "src/cpp_knn/verify.py"}
SEMANTIC_CONFIG_FILES = {"configs/baseline.json", "configs/final.json"}
STABLE_ROOT_OUTPUTS = {"resolved_configs.json", "release_resolved_configs.json", "data_manifest.json",
                       "environment.json", "metrics.csv", "agreement.csv"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scientific_hashes(hashes):
    """Guard executable science; frozen configs are checked by their model semantic hashes.

    The two config files may gain descriptive metadata outside ``model``. Their
    original raw hashes remain in the inference manifest, while release_sources
    records current bytes. Other config files remain guarded by raw hashes.
    """
    return {name: digest for name, digest in hashes.items()
            if name not in ARCHIVED_CPP_HELPERS | SEMANTIC_CONFIG_FILES
            and (name in SCIENTIFIC_FILES or name.startswith(SCIENTIFIC_PREFIXES))}


def reusable_run(manifest, current_hashes, configs, data_hash, saved_data_manifest):
    """``configs`` must come from load_configs(), which verifies each full model hash."""
    reasons = []
    if manifest.get("status") not in {"complete", "complete_with_documented_differences"}:
        reasons.append("Run is not complete")
    if scientific_hashes(manifest.get("sources", {})) != scientific_hashes(current_hashes):
        reasons.append("Prediction, preprocessing, build or benchmark source hashes changed")
    if manifest.get("config_hashes") != {key: value["config_hash"] for key, value in configs.items()}:
        reasons.append("Frozen configuration hashes changed")
    if saved_data_manifest.get("dataset", {}).get("sha256") != data_hash:
        reasons.append("Dataset hash changed")
    if not manifest.get("benchmark_requested"):
        reasons.append("Run has no completed benchmark request")
    return not reasons, reasons


def stable_output_hashes(run_dir):
    """Hash saved scientific output bytes, avoiding metadata/report self-reference.

    Prepared/native files already have their own data/build provenance and are
    deliberately omitted from the compact submission. Report/validation files
    are independently covered by the submission ZIP byte manifest.
    """
    run_dir = Path(run_dir)
    selected = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        if (relative.parts[0] in {"predictions", "diagnostics"}
                or len(relative.parts) == 1 and (path.name in STABLE_ROOT_OUTPUTS
                                                 or path.name.startswith("benchmark_") and path.suffix in {".csv", ".json"})):
            selected.append((relative.as_posix(), path))
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in selected}


def verify_recorded_outputs(run_dir, manifest):
    """Once scientific outputs have been sealed, later release metadata cannot replace them."""
    for name, expected in manifest.get("outputs", {}).items():
        # This single supplement may acquire more non-model metadata on a later release pass.
        if name == "release_resolved_configs.json":
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe recorded output path")
        actual = hashlib.sha256((Path(run_dir) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Previously sealed scientific output changed: {name}")


def finalize_release_manifest(run_dir, release_sources, configs, validated_evidence):
    """Attach release provenance after validation without rewriting inference provenance.

    This never computes predictions, timings or metrics. Original ``sources`` and
    ``resolved_configs.json`` retain the bytes/schema used for the measured run.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") not in {"complete", "complete_with_documented_differences"}:
        raise ValueError("Only a completed, validated run can receive release provenance")
    if validated_evidence.get("run_id") != manifest.get("run_id"):
        raise ValueError("Validation evidence belongs to a different run")
    if scientific_hashes(manifest.get("sources", {})) != scientific_hashes(release_sources):
        raise ValueError("Scientific sources changed after the measured run")
    verify_recorded_outputs(run_dir, manifest)
    original_configs = read_json(run_dir / "resolved_configs.json")
    if set(configs) != set(original_configs):
        raise ValueError("Release model groups differ from the measured run")
    metadata_changes = {}
    for group, config in configs.items():
        original = original_configs[group]
        for key in ("model", "config_hash", "schema_version", "model_group"):
            if config.get(key) != original.get(key):
                raise ValueError(f"Release finalization cannot alter measured model configuration: {group}/{key}")
        model_hash = hashlib.sha256(json.dumps(config["model"], sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()
        if config["config_hash"] != model_hash or manifest["config_hashes"].get(group) != model_hash:
            raise ValueError("Release config hash does not describe the measured model")
        metadata_changes[group] = sorted(key for key in set(config) | set(original)
                                         if config.get(key) != original.get(key))
    current_prediction_hashes = {Path(name).stem: digest for name, digest in stable_output_hashes(run_dir).items()
                                if name.startswith("predictions/") and name.endswith(".csv")}
    if current_prediction_hashes != validated_evidence.get("prediction_sha256"):
        raise ValueError("Prediction files changed after saved-run validation")
    release_configs_path = run_dir / "release_resolved_configs.json"
    write_json(release_configs_path, configs)
    manifest["release_sources"] = release_sources
    manifest["outputs"] = stable_output_hashes(run_dir)
    manifest["release_provenance"] = {
        "finalized_at": now(),
        "scope": "Release metadata supplement after validation; no inference or benchmark repeated",
        "inference_sources_preserved": "sources records the original measured-run raw source hashes",
        "inference_configuration_preserved": "resolved_configs.json remains the original measured configuration",
        "release_configuration": release_configs_path.name,
        "metadata_only_configuration_changes": metadata_changes,
        "configuration_guard": "Full model dictionaries and semantic config hashes unchanged; only external metadata may differ",
        "output_hash_algorithm": "SHA-256 of exact saved bytes, paths relative to this run",
        "output_hash_scope": "Predictions, metrics, agreement, benchmark evidence, diagnostics and fixed data/environment/config manifests",
        "output_hash_exclusions": ["run_manifest.json (self-reference)", "validation.json", "release_validation_external.json",
                                   "report_manifest.json", "summary.md", "figures/**", "data/**", "native/**"],
        "excluded_output_evidence": "Report/validation bytes are checked by the exported ZIP manifest; prepared data files are hashed in data_manifest.json",
    }
    write_json(manifest_path, manifest)
    return {"run_id": manifest["run_id"], "output_files": len(manifest["outputs"]),
            "metadata_only_configuration_changes": metadata_changes,
            "release_source_files": len(release_sources), "original_inference_provenance_preserved": True}


def scope_status(checks, scope):
    statuses = {entry["status"] for entry in checks if entry["scope"] == scope}
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if not statuses or "SKIP" in statuses:
        return "SKIP"
    return "PASS"


def completion(checks, *, full, differences=False):
    levels = {scope: scope_status(checks, scope) for scope in SCOPES}
    if not full or any(value != "PASS" for value in levels.values()):
        return "partial", levels
    return ("complete_with_documented_differences" if differences else "complete"), levels


def portable_text(text, root=ROOT):
    value = str(text).replace("\\", "/")
    value = value.replace(str(root).replace("\\", "/"), "{project}")
    value = value.replace(sys.executable.replace("\\", "/"), "python")
    # Keep portable command evidence while full subprocess logs remain local.
    value = re.sub(r"[A-Za-z]:/+Users/+[^/\s\"']+", "{user}", value)
    value = re.sub(r"/(?:home|Users)/[^/\s\"']+", "{user}", value)
    return value


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return portable_text(value) if isinstance(value, (str, Path)) else value


class Session:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.logs = self.directory / "logs"
        self.logs.mkdir()
        self.checks = []

    def record(self, check_id, scope, status, reason="", command=(), log=None, started=None, exit_code=None, evidence=None):
        row = {"check_id": check_id, "scope": scope, "command": list(command), "started": started or now(),
               "finished": now(), "exit_code": exit_code, "status": status, "reason": reason,
               "log": str(log) if log else None}
        if evidence is not None:
            row["evidence"] = evidence
        self.checks.append(portable(row))
        write_json(self.directory / "checks.json", self.checks)
        print(f"{check_id}: {status}" + (f" ({reason})" if reason else ""), flush=True)
        return row

    def command(self, check_id, scope, command, *, timeout=1200, environment_updates=None):
        from src.weka_bridge.runner import _kill_tree
        started = now()
        log = self.logs / f"{check_id}.log"
        status, reason, exit_code = "PASS", "", None
        environment = os.environ.copy()
        environment.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        if environment_updates:
            environment.update(environment_updates)
        try:
            with log.open("w", encoding="utf-8") as output:
                with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output,
                                      stderr=subprocess.STDOUT, start_new_session=(os.name != "nt")) as process:
                    try:
                        exit_code = process.wait(timeout=timeout)
                    except BaseException:
                        _kill_tree(process)
                        process.wait(timeout=10)
                        raise
            if exit_code:
                status, reason = "FAIL", "Nonzero exit; see local command log"
                text = log.read_text(encoding="utf-8", errors="replace").lower()
                if check_id.startswith("cpp_"):
                    prefix = check_id.replace("_sanitized", "-sanitized")
                    for nested_log in self.logs.glob(prefix + "_*.log"):
                        text += nested_log.read_text(encoding="utf-8", errors="replace").lower()
                if check_id == "weka_clean_build_tests":
                    nested_log = ROOT / "weka/target/maven_build.log"
                    if nested_log.is_file():
                        text += nested_log.read_text(encoding="utf-8", errors="replace").lower()
                markers = ("permission denied", "access is denied", "getsockopt", "could not resolve",
                           "maven unavailable:", "java unavailable;", "cmake is unavailable.", "no cmake_cxx_compiler", "0xc0000142",
                           "failed to initialize", "could not find a version", "temporary failure")
                if any(marker in text for marker in markers):
                    status, reason = "BLOCKED", "Environment/toolchain/runtime access blocked this command; see local log"
            elif "unit_tests" in check_id:
                text = log.read_text(encoding="utf-8", errors="replace")
                skipped = re.search(r"skipped=(\d+)", text)
                if skipped and int(skipped.group(1)):
                    status, reason = "SKIP", f"Unit suite skipped {skipped.group(1)} tests/classes; see verbose log"
        except FileNotFoundError as error:
            status, reason = "BLOCKED", str(error)
        except subprocess.TimeoutExpired:
            status, reason = "BLOCKED", f"Command exceeded {timeout} seconds; child process tree terminated"
        except KeyboardInterrupt:
            self.record(check_id, scope, "BLOCKED", "Interrupted; child process tree terminated", command, log, started)
            raise
        return self.record(check_id, scope, status, reason, command, log, started, exit_code)

    def python(self, check_id, scope, code, timeout=1200):
        return self.command(check_id, scope, [sys.executable, "-c", code], timeout=timeout)


def validate_saved_run(run_dir):
    """Recompute all contract/metric/benchmark checks from this run's actual files."""
    import numpy as np
    import pandas as pd
    from src.common.experiment import IMPLEMENTATIONS, SCORE_TOLERANCE, array_hash, model_group, sha256
    from src.evaluation.final import read_predictions, validate_predictions, metrics
    from src.benchmarking.suite import summarize

    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    verify_recorded_outputs(run_dir, manifest)
    configs = read_json(run_dir / "resolved_configs.json")
    data_manifest = read_json(run_dir / "data_manifest.json")
    if set(manifest.get("completed_implementations", [])) != set(IMPLEMENTATIONS):
        raise ValueError("The complete five-model comparison is required")
    characterization = read_json(ROOT / "docs/migration_characterization.json")
    for group in ("baseline", "final"):
        details = data_manifest["groups"][group]
        for name, digest in details["files"].items():
            if sha256(run_dir / "data" / group / name) != digest:
                raise ValueError(f"Prepared file hash changed: {group}/{name}")
        for partition in ("train", "test"):
            with np.load(run_dir / "data" / group / f"{partition}.npz", allow_pickle=False) as prepared:
                matrix = prepared["X"]
                if array_hash(matrix) != details[f"{partition}_numeric_hash"]:
                    raise ValueError("Prepared numeric matrix differs from the manifest")
                raw_hash = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
                if raw_hash != characterization[group][f"{partition}_raw_array_sha256"]:
                    raise ValueError(f"Pre-migration matrix regression: {group}/{partition}")
                if prepared["feature_names"].tolist() != characterization[group]["feature_names"]:
                    raise ValueError("Pre-migration feature ordering changed")
    frames = {}
    for name in IMPLEMENTATIONS:
        frames[name] = validate_predictions(read_predictions(run_dir / "predictions" / f"{name}.csv"),
                                           manifest["run_id"], name, configs[model_group(name)], data_manifest)
        if len(frames[name]) != 6000:
            raise ValueError("Expected all 6000 held-out rows")
    for name, group in (("baseline_python_v5_1", "baseline"), ("final_python", "final")):
        score_hash = hashlib.sha256(frames[name].score_class_1.to_numpy(dtype=np.float64).tobytes()).hexdigest()
        if score_hash != characterization[group]["scores_raw_array_sha256"]:
            raise ValueError(f"Pre-migration all-row score regression: {name}")
    stored = pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip").set_index("implementation_id")
    for name, frame in frames.items():
        actual = metrics(frame)
        for column, value in actual.items():
            if isinstance(value, (int, float, np.number)) and not np.isclose(stored.loc[name, column], value, atol=1e-12, rtol=0):
                raise ValueError(f"Stored metric differs: {name}/{column}")
        tn, fp, fn, tp = [actual[key] for key in ("TN", "FP", "FN", "TP")]
        # Arithmetic independent of sklearn metric helpers.
        arithmetic = {"accuracy": (tn + tp) / (tn + fp + fn + tp),
                      "precision": tp / (tp + fp) if tp + fp else 0,
                      "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}
        for key, value in arithmetic.items():
            if not np.isclose(value, actual[key], atol=1e-12, rtol=0):
                raise ValueError(f"Confusion arithmetic differs: {name}/{key}")
    left, right = frames["final_python"], frames["final_cpp"]
    if not np.array_equal(left.y_pred, right.y_pred) or np.max(np.abs(left.score_class_1 - right.score_class_1)) > SCORE_TOLERANCE:
        raise ValueError("Manual Python/C++ identical-semantics agreement failed")
    # Every pair, including external toolkits, must be honestly represented in the stored table.
    agreements = pd.read_csv(run_dir / "agreement.csv", float_precision="round_trip")
    if len(agreements) != 6:
        raise ValueError("Expected all six final-model pairs")
    for row in agreements.itertuples():
        a, b = frames[row.implementation_a], frames[row.implementation_b]
        delta = np.abs(a.score_class_1.to_numpy() - b.score_class_1.to_numpy())
        if row.label_disagreements != int(np.count_nonzero(a.y_pred != b.y_pred)):
            raise ValueError("Stored toolkit label agreement differs")
        if row.scores_over_tolerance != int(np.count_nonzero(delta > SCORE_TOLERANCE)):
            raise ValueError("Stored score-difference count differs")
        if not np.isclose(row.max_abs_score_difference, delta.max(), atol=1e-15, rtol=0):
            raise ValueError("Stored max score difference differs")
    raw = pd.read_csv(run_dir / "benchmark_raw.csv", float_precision="round_trip")
    if raw.empty or raw.sequence.duplicated().any() or not np.isfinite(raw.seconds).all() or (raw.seconds <= 0).any():
        raise ValueError("Missing or invalid raw runtime evidence")
    for name in IMPLEMENTATIONS:
        subset = raw[(raw.implementation_id == name) & (raw.phase == "prediction")]
        if len(subset[~subset.is_warmup]) != 20 or len(subset[subset.is_warmup]) != 3:
            raise ValueError("Full release evidence requires 20 timed trials and 3 warmups per implementation")
        if subset.process_id.nunique() != 1 or subset.checksum.nunique() != 1:
            raise ValueError("Warmups/timed predictions changed process or prediction checksum")
        if not (subset.n_train == 24000).all() or not (subset.n_query == 6000).all():
            raise ValueError("Benchmark did not use the full workload")
        expected = configs[model_group(name)]["config_hash"]
        if not (subset.config_hash == expected).all():
            raise ValueError("Benchmark config identity differs")
        if len(raw[(raw.implementation_id == name) & (raw.phase == "prepared_pipeline")]) != 5:
            raise ValueError("Expected five symmetric fresh-process trials per implementation")
    summary = summarize(raw).sort_values(["implementation_id", "phase"]).reset_index(drop=True)
    saved = pd.read_csv(run_dir / "benchmark_summary.csv", float_precision="round_trip").sort_values(["implementation_id", "phase"]).reset_index(drop=True)
    for column in summary.select_dtypes(include="number"):
        if not np.allclose(summary[column], saved[column], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f"Benchmark summary differs: {column}")
    predictions = {name: sha256(run_dir / "predictions" / f"{name}.csv") for name in IMPLEMENTATIONS}
    return {"rows_per_implementation": 6000, "implementations": list(IMPLEMENTATIONS),
            "prediction_sha256": predictions, "manual_cpp_max_score_difference": float(np.max(np.abs(left.score_class_1 - right.score_class_1))),
            "external_numerical_differences": bool((agreements.scores_over_tolerance > 0).any()),
            "benchmark_rows": len(raw), "timed_prediction_trials": 20, "warmups": 3,
            "prepared_pipeline_trials": 5, "run_id": manifest["run_id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--quick", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args(argv)
    if args.full and (args.data is None or not args.data.is_file()):
        parser.error("--full requires an existing --data file")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    session = Session(args.work_root or ROOT / "verification" / f"release_{tag}")
    run_dir = None
    evidence = {}
    session.command("compile", "build correctness", [sys.executable, "-m", "compileall", "-q", "src", "run_all.py"])
    if args.quick:
        session.command("active_unit_tests", "build correctness", [sys.executable, "-m", "unittest", "discover", "-v", "-s", "tests"])
    session.command("archive_bytes", "packaging completeness", [sys.executable, "scripts/verify_archive.py", "--output", str(session.directory / "archive.json")])
    from src.common.experiment import BASELINE_SHA256, environment, load_configs, sha256, source_hashes
    baseline_ok = sha256(ROOT / "src/custom_knn/classifier.py") == BASELINE_SHA256
    session.record("baseline_source", "same-semantics numerical correctness", "PASS" if baseline_ok else "FAIL",
                   evidence={"expected_sha256": BASELINE_SHA256, "actual_sha256": sha256(ROOT / "src/custom_knn/classifier.py")})
    if args.full:
        actual_environment = environment()
        thread_counts = [pool.get("num_threads") for pool in actual_environment["threadpools"]]
        session.record("environment_preflight", "build correctness", "PASS" if all(count == 1 for count in thread_counts) else "FAIL",
                       "Thread limits set before numeric imports; no OS affinity or priority changes",
                       evidence=actual_environment)
        try:
            configs = load_configs()
            session.record("frozen_configuration", "build correctness", "PASS", evidence={key: value["config_hash"] for key, value in configs.items()})
        except Exception as error:
            configs = None
            session.record("frozen_configuration", "build correctness", "FAIL", str(error))
        portable_executable = None
        for mode, sanitize in (("portable", False), ("native", False), ("portable", True)):
            name = f"cpp_{mode}" + ("_sanitized" if sanitize else "")
            code = ("from src.cpp_knn.bridge import build; "
                    f"print(build({mode!r},sanitize={sanitize!r},log_dir={str(session.logs)!r},"
                    f"build_dir={str(session.directory / 'build' / name)!r}))")
            built = session.python(name, "optional sanitizer coverage" if sanitize else "build correctness", code)
            if sanitize and built["status"] == "FAIL":
                # An actual sanitizer-detected defect must still fail the release.
                session.record("sanitizer_detected_failure", "build correctness", "FAIL", "Sanitizer failed without a recognized environment/runtime blocker; inspect its log")
            if mode == "portable" and not sanitize and built["status"] == "PASS":
                from src.cpp_knn.bridge import executable_path
                portable_executable = executable_path("portable", session.directory / "build" / name)
        from src.cpp_knn.bridge import find_cmake
        try:
            cmake = find_cmake()
            debug_directory = session.directory / "build/cpp_debug"
            configure = [str(cmake), "-S", str(ROOT), "-B", str(debug_directory), "-DCMAKE_BUILD_TYPE=Debug",
                         "-DBUILD_TESTING=ON", "-DCPP_KNN_NATIVE=OFF", "-DCPP_KNN_SANITIZE=OFF"]
            if os.name == "nt" and "Microsoft Visual Studio" in str(cmake):
                configure += ["-G", "Visual Studio 17 2022", "-A", "x64"]
            if session.command("cpp_debug_configure", "build correctness", configure)["status"] == "PASS":
                if session.command("cpp_debug_build", "build correctness", [str(cmake), "--build", str(debug_directory), "--config", "Debug"])["status"] == "PASS":
                    ctest = cmake.with_name("ctest.exe" if os.name == "nt" else "ctest")
                    session.command("cpp_debug_ctest", "build correctness", [str(ctest), "--test-dir", str(debug_directory), "-C", "Debug", "--output-on-failure"])
        except (OSError, RuntimeError) as error:
            session.record("cpp_debug_toolchain", "build correctness", "BLOCKED", str(error))
        session.python("weka_clean_build_tests", "build correctness", "from src.weka_bridge.runner import build; print(build(force=True))")
        session.command("active_unit_tests", "build correctness", [sys.executable, "-m", "unittest", "discover", "-v", "-s", "tests"],
                        environment_updates={"CPP_KNN_TEST_EXECUTABLE": str(portable_executable)} if portable_executable else None)
        # Historical unit results may be reused only when tied to the unchanged archive manifest.
        from scripts.verify_archive import verify_snapshot
        historical = ROOT / "verification/historical_release_check/historical_validation.json"
        snapshot = ROOT / "archive/research/pre_final_df1a2c3f67e4"
        old = read_json(historical) if historical.is_file() else {}
        current_archive = verify_snapshot(snapshot)
        if old.get("status") == "PASS" and old.get("snapshot_after", {}).get("manifest_sha256") == current_archive["manifest_sha256"] and current_archive["status"] == "PASS":
            session.record("historical_unit_tests", "packaging completeness", "PASS", "Reused verified historical suite; no expensive historical model search", evidence={"proof": str(historical), "archive_manifest_sha256": current_archive["manifest_sha256"]})
        else:
            session.command("historical_unit_tests", "packaging completeness", [sys.executable, "scripts/reproduce_research.py", "--snapshot", str(snapshot), "--work-root", str(session.directory / "historical")])
        if configs:
            candidates = [args.run_dir] if args.run_dir else sorted((ROOT / "results/final").glob("*"), reverse=True)
            for candidate in candidates:
                try:
                    manifest = read_json(candidate / "run_manifest.json")
                    current, reasons = reusable_run(manifest, source_hashes(), configs, sha256(args.data), read_json(candidate / "data_manifest.json"))
                    if current:
                        evidence = validate_saved_run(candidate)
                        run_dir = candidate.resolve()
                        session.record("reuse_current_benchmark", "runtime evidence", "PASS", "Sources/configuration/data and all prediction/benchmark evidence match; no repeated benchmark", evidence=evidence)
                        break
                    if args.run_dir:
                        session.record("requested_run_reuse", "provenance selection", "SKIP", "; ".join(reasons))
                except (OSError, ValueError, KeyError) as error:
                    if args.run_dir:
                        session.record("requested_run_reuse", "provenance selection", "SKIP", str(error))
            if run_dir is None and args.run_dir:
                session.record("selected_run_invalid", "runtime evidence", "FAIL",
                               "Explicit --run-dir could not be verified; preserving its evidence and refusing an automatic benchmark replacement")
            if run_dir is None and not args.run_dir:
                output = ROOT / "results/final"
                before_runs = set(output.glob("*/run_manifest.json"))
                result = session.command("five_model_integration_benchmark", "runtime evidence", [sys.executable, "run_all.py", "--data", str(args.data.resolve()), "--build", "--benchmark", "--runs", "20", "--warmups", "3", "--output-root", str(output)], timeout=7200)
                if result["status"] == "PASS":
                    found = list(set(output.glob("*/run_manifest.json")) - before_runs)
                    if len(found) == 1:
                        run_dir = found[0].parent
                        try:
                            evidence = validate_saved_run(run_dir)
                        except Exception as error:
                            session.record("saved_run_contract", "real-data reproduction", "FAIL", str(error))
                            run_dir = None
            if run_dir:
                session.record("all_row_contract_metrics", "real-data reproduction", "PASS", evidence=evidence)
                session.record("manual_python_cpp", "same-semantics numerical correctness", "PASS", evidence={"rows": 6000, "max_absolute_score_difference": evidence["manual_cpp_max_score_difference"], "tolerance": 1e-12})
                session.record("genuine_toolkits", "toolkit comparability", "PASS", "All rows from genuine sklearn and Weka; native differences retained", evidence={"differences_observed": evidence["external_numerical_differences"]})
                investigation = run_dir / "diagnostics/numerical_investigation.json"
                if investigation.is_file():
                    investigation_data = read_json(investigation)
                    matching = (investigation_data.get("run_id") == evidence["run_id"]
                                and investigation_data.get("config_hash") == configs["final"]["config_hash"])
                    session.record("numerical_investigation", "toolkit comparability", "PASS" if matching else "FAIL",
                                   "Preserved existing investigation with checked run/config identity; existing output seals are verified before reuse",
                                   evidence={"path": str(investigation), "sha256": sha256(investigation), "identity_matches": matching})
                else:
                    session.command("numerical_investigation", "toolkit comparability",
                                    [sys.executable, "-m", "src.evaluation.diagnostics", "--run-dir", str(run_dir)])
                session.command("report_regeneration", "real-data reproduction", [sys.executable, "-m", "src.reporting", "--run-dir", str(run_dir)])
                session.command("clean_working_tree", "packaging completeness", [sys.executable, "scripts/verify_clean_checkout.py", "--data", str(args.data.resolve()), "--work-root", str(session.directory / "clean_source"), "--reference-run", str(run_dir)], timeout=3600)
                source_status, source_scopes = completion(session.checks, full=True, differences=evidence.get("external_numerical_differences", False))
                compact = {"status": source_status, "full_validation": True, "completion_scopes": source_scopes,
                           "checks": session.checks, "session_report": portable_text(session.directory / "validation.json"),
                           "zip_verification": "Actual ZIP extraction/build/run proof is external to avoid a circular ZIP self-hash; see session report"}
                write_json(run_dir / "validation.json", compact)
                try:
                    finalized = finalize_release_manifest(run_dir, source_hashes(), load_configs(), evidence)
                    session.record("release_output_provenance", "packaging completeness", "PASS",
                                   "Original inference sources/config preserved; stable outputs hashed and release metadata attached", evidence=finalized)
                except (OSError, ValueError, KeyError) as error:
                    session.record("release_output_provenance", "packaging completeness", "FAIL", str(error))
                    finalized = None
                compact["status"], compact["completion_scopes"] = completion(
                    session.checks, full=True, differences=evidence.get("external_numerical_differences", False))
                write_json(run_dir / "validation.json", compact)
                zip_path = ROOT / "dist/3804ICT-final-submission.zip"
                history_zip = ROOT / "dist/3804ICT-research-history.zip"
                exported = (session.command("submission_export", "packaging completeness", [sys.executable, "scripts/export_submission.py", "--output", str(zip_path), "--run-dir", str(run_dir), "--history-output", str(history_zip)])
                            if finalized else session.record("submission_export", "packaging completeness", "BLOCKED", "Release provenance finalization failed"))
                if exported["status"] == "PASS":
                    session.command("clean_submission_zip", "packaging completeness", [sys.executable, "scripts/verify_clean_checkout.py", "--data", str(args.data.resolve()), "--work-root", str(session.directory / "clean_zip"), "--source-zip", str(zip_path), "--reference-run", str(run_dir)], timeout=3600)
                    evidence["submission_sha256"] = sha256(zip_path)
                    evidence["history_sha256"] = sha256(history_zip)
        if run_dir is None:
            for scope in ("real-data reproduction", "toolkit comparability", "runtime evidence", "packaging completeness"):
                session.record("required_run_unavailable_" + scope.replace(" ", "_"), scope, "BLOCKED", "No verified complete five-model benchmark run is available")
    else:
        for scope in SCOPES[2:]:
            session.record("quick_omits_" + scope.replace(" ", "_"), scope, "SKIP", "Dataset-free quick checks do not replace full release validation")
    status, levels = completion(session.checks, full=args.full, differences=evidence.get("external_numerical_differences", False))
    report = {"schema_version": 1, "status": status, "full_validation": args.full,
              "completion_scopes": levels, "checks": session.checks, "run_dir": portable_text(run_dir) if run_dir else None,
              "evidence": evidence, "limitations": ["Previously inspected test; no new independent ML validation.",
              "Library-native score and tie rules are comparisons, not identical-semantics oracles.",
              "Raw build/subprocess logs remain local; exported command paths use portable placeholders."]}
    write_json(session.directory / "validation.json", portable(report))
    if run_dir:
        write_json(run_dir / "release_validation_external.json", portable(report))
    print(json.dumps({"status": status, "completion_scopes": levels, "report": portable_text(session.directory / "validation.json")}, indent=2))
    return 0 if (status.startswith("complete") or (args.quick and not any(row["status"] in {"FAIL", "BLOCKED"} for row in session.checks))) else 1
