"""Build and execute current sources in a fresh tree with a fresh virtual environment."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

try:
    from .export_submission import REQUIRED_IMPLEMENTATIONS, ROOT, source_files, verify_zip
    from .verify_archive import safe_relative
except ImportError:
    from export_submission import REQUIRED_IMPLEMENTATIONS, ROOT, source_files, verify_zip
    from verify_archive import safe_relative


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_command(check_id: str, command: list[str], tree: Path, logs: Path,
                   timeout: int = 1200, environment: dict | None = None) -> dict:
    started = timestamp()
    log = logs / f"{check_id}.log"
    status, reason, exit_code = "PASS", "", None
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["PYTHONNOUSERSITE"] = "1"
    if environment:
        env.update(environment)
    try:
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=tree, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout, env=env, check=False)
        exit_code = completed.returncode
        if exit_code:
            status, reason = "FAIL", "Command returned a nonzero exit code; see log"
            output = log.read_text(encoding="utf-8", errors="replace")
            unavailable = ("No module named pip", "Could not find a version", "Temporary failure in name resolution",
                           "Failed to establish a new connection", "Network is unreachable", "not found on PATH",
                           "Could NOT find", "No CMAKE_CXX_COMPILER", "not recognized", "No such file or directory")
            if any(marker.lower() in output.lower() for marker in unavailable):
                status, reason = "BLOCKED", "Required dependency/toolchain could not be obtained or found; see log"
    except FileNotFoundError as error:
        status, reason = "BLOCKED", f"Required executable unavailable: {Path(command[0]).name}"
        log.write_text(str(error) + "\n", encoding="utf-8")
    except subprocess.TimeoutExpired:
        status, reason = "BLOCKED", f"Command exceeded {timeout} seconds"
    output = log.read_text(encoding="utf-8", errors="replace")
    test_count = re.search(r"Ran (\d+) tests?", output)
    skip_count = re.search(r"skipped=(\d+)", output)
    if status == "PASS" and skip_count and int(skip_count.group(1)):
        status, reason = "SKIP", "The unit suite skipped checks; verbose log retains each reason"
    print(f"{check_id}: {status}", flush=True)
    return {"check_id": check_id, "scope": "fresh-tree reproduction", "command": command,
            "started": started, "finished": timestamp(), "exit_code": exit_code,
            "status": status, "reason": reason, "log": log.relative_to(logs.parent).as_posix(),
            "tests_run": int(test_count.group(1)) if test_count else None,
            "tests_skipped": int(skip_count.group(1)) if skip_count else 0}


def prepare_tree(root: Path, tree: Path, source_zip: Path | None = None) -> dict:
    if tree.exists():
        raise FileExistsError("Disposable tree already exists; use a new --work-root")
    tree.mkdir(parents=True)
    if source_zip:
        verification = verify_zip(source_zip)
        if verification["kind"] != "final-submission":
            raise ValueError("Clean ZIP check requires a final-submission archive")
        with zipfile.ZipFile(source_zip) as archive:
            for info in archive.infolist():
                destination = tree / safe_relative(info.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info.filename))
        return {"mode": "submission ZIP extraction", "zip": verification}
    files = source_files(root)
    for relative, path in files.items():
        destination = tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return {"mode": "current working tree copy", "files": len(files),
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}


def compare_reproduction(reference_run: Path, child_run: Path) -> dict:
    """Compare saved results and canonical matrix identities, without timing or inference."""
    tolerance = 1e-12
    statuses = {"complete", "completed", "complete_with_documented_differences"}
    manifests = [json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
                 for run in (reference_run, child_run)]
    errors = []
    for role, manifest in zip(("reference", "child"), manifests):
        if manifest.get("status") not in statuses:
            errors.append(f"{role} run is not a completed five-implementation run")
    if manifests[0].get("config_hashes") != manifests[1].get("config_hashes"):
        errors.append("Frozen configuration hashes differ")
    data = [json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
            for run in (reference_run, child_run)]
    if data[0]["dataset"]["sha256"] != data[1]["dataset"]["sha256"]:
        errors.append("Input dataset byte hashes differ")
    if data[0]["split"] != data[1]["split"]:
        errors.append("Split identities, row order or ground-truth hashes differ")
    groups = {}
    fields = ("config_hash", "training_shape", "test_shape", "dtype", "feature_names", "train_numeric_hash", "test_numeric_hash")
    for group in ("baseline", "final"):
        matches = {field: data[0]["groups"][group][field] == data[1]["groups"][group][field] for field in fields}
        groups[group] = {"status": "PASS" if all(matches.values()) else "FAIL", "equal_fields": matches}
        if not all(matches.values()):
            errors.append(f"{group} canonical matrices or feature order differ")
    comparisons = {}
    identities = ("test_position", "row_id", "original_id", "y_true", "implementation_id", "model_group", "config_hash", "decision_rule", "threshold")
    for implementation in sorted(REQUIRED_IMPLEMENTATIONS):
        rows = []
        headers = []
        for run in (reference_run, child_run):
            with (run / "predictions" / f"{implementation}.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                headers.append(reader.fieldnames)
                rows.append(list(reader))
        if headers[0] != headers[1] or not set((*identities, "run_id", "y_pred", "score_class_1")).issubset(headers[0] or []):
            raise ValueError(f"Prediction schemas differ or required columns are missing: {implementation}")
        identity_mismatches = label_differences = above_tolerance = exact_labels = 0
        maximum_difference = 0.0
        for position, (reference, child) in enumerate(zip(*rows)):
            aligned = (all(reference[name] == child[name] for name in identities)
                       and reference["test_position"] == str(position)
                       and reference["run_id"] == manifests[0]["run_id"]
                       and child["run_id"] == manifests[1]["run_id"])
            identity_mismatches += not aligned
            same_label = reference["y_pred"] == child["y_pred"]
            label_differences += not same_label
            exact_labels += aligned and same_label
            probabilities = [float(row["score_class_1"]) for row in (reference, child)]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
                raise ValueError(f"Nonfinite/out-of-range score: {implementation}, position {position}")
            delta = abs(probabilities[0] - probabilities[1])
            maximum_difference = max(maximum_difference, delta)
            above_tolerance += delta > tolerance
        passed = len(rows[0]) == len(rows[1]) > 0 and not (identity_mismatches or label_differences or above_tolerance)
        comparisons[implementation] = {"status": "PASS" if passed else "FAIL", "reference_rows": len(rows[0]),
            "child_rows": len(rows[1]), "identity_mismatches": identity_mismatches,
            "exact_label_matches": exact_labels, "label_disagreements": label_differences,
            "scores_over_tolerance": above_tolerance, "max_absolute_score_difference": maximum_difference}
        if not passed:
            errors.append(f"Saved prediction reproduction differs: {implementation}")
    return {"status": "FAIL" if errors else "PASS", "scope": "All five saved score/label/identity vectors and both canonical matrix representations; timings excluded",
            "reference_run_id": manifests[0]["run_id"], "child_run_id": manifests[1]["run_id"],
            "score_tolerance": tolerance, "matrices": groups, "implementations": comparisons, "errors": errors}


def verify_clean(root: Path, data: Path, work_root: Path, source_zip: Path | None = None,
                 reference_run: Path | None = None) -> dict:
    root, data, work_root = root.resolve(), data.resolve(), work_root.resolve()
    if not data.is_file():
        raise FileNotFoundError("External dataset does not exist")
    if work_root == root or root.is_relative_to(work_root) or work_root.is_relative_to(root / "archive"):
        raise ValueError("Work root must be a new disposable directory, outside the immutable archive")
    if work_root.exists():
        raise FileExistsError("Work root exists; use a new disposable directory")
    work_root.mkdir(parents=True)
    logs = work_root / "logs"
    logs.mkdir()
    tree = work_root / "tree"
    prepared = prepare_tree(root, tree, source_zip)
    forbidden = [p for p in (".venv", ".git", "build", "weka/target", "data/processed") if (tree / p).exists()]
    if forbidden:
        raise ValueError(f"Clean tree unexpectedly contains generated directories: {forbidden}")
    requirements = [line.strip() for line in (tree / "requirements.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", line) for line in requirements):
        raise ValueError("Clean reproduction requires exact pinned requirements")
    checks = []
    report = {"schema_version": 1, "status": "started", "source": prepared,
              "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
              "initial_generated_directories": forbidden, "checks": checks,
              "scope": "Fresh venv, install, pip check, tests, native builds, five-model integration, report regeneration. No recursive validation or repeated benchmark."}
    report_path = work_root / "clean_validation.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def execute(check_id, command, timeout=1200):
        record = record_command(check_id, command, tree, logs, timeout)
        checks.append(record)
        save()
        return record["status"] == "PASS"

    save()
    ready = execute("create_venv", [sys.executable, "-m", "venv", ".venv"])
    python = tree / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if ready:
        ready = execute("install_pinned_dependencies", [str(python), "-m", "pip", "install",
                         "--disable-pip-version-check", "-r", "requirements.txt"])
    if ready:
        ready = execute("pip_check", [str(python), "-m", "pip", "check"])
    if ready:
        execute("compile", [str(python), "-m", "compileall", "-q", "src", "run_all.py"])
        output_root = "results/clean_reproduction"
        ready = execute("five_model_build_and_run", [str(python), "run_all.py", "--data", str(data),
                        "--build", "--output-root", output_root], timeout=1800)
        # Native CLI test classes discover the freshly built binary here. Running
        # only before the build would silently skip the clean CLI integration.
        execute("unit_tests", [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"])
        if ready:
            manifests = sorted((tree / output_root).glob("*/run_manifest.json"))
            if len(manifests) != 1:
                checks.append({"check_id": "completed_run", "scope": "fresh-tree reproduction", "command": [],
                               "started": timestamp(), "finished": timestamp(), "exit_code": 1, "status": "FAIL",
                               "reason": "Expected exactly one new final run", "log": None})
            else:
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                status = "PASS" if manifest.get("status") in {"complete", "completed", "complete_with_documented_differences"} else "FAIL"
                checks.append({"check_id": "completed_run", "scope": "fresh-tree reproduction", "command": [],
                               "started": timestamp(), "finished": timestamp(), "exit_code": 0 if status == "PASS" else 1,
                               "status": status, "reason": manifest.get("status"),
                               "log": manifests[0].relative_to(work_root).as_posix()})
                execute("regenerate_reports", [str(python), "-m", "src.reporting", "--run-dir",
                        str(manifests[0].parent.relative_to(tree))])
                report["run_manifest"] = manifests[0].relative_to(work_root).as_posix()
                if reference_run is not None:
                    started = timestamp()
                    try:
                        comparison = compare_reproduction(reference_run.resolve(), manifests[0].parent)
                    except (OSError, KeyError, TypeError, ValueError) as error:
                        comparison = {"status": "FAIL", "errors": [str(error)]}
                    report["reference_reproduction"] = comparison
                    (logs / "reference_reproduction.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
                    checks.append({"check_id": "reference_reproduction", "scope": "All five predictions, canonical matrix hashes and feature order", "command": [],
                                   "started": started, "finished": timestamp(), "exit_code": 0 if comparison["status"] == "PASS" else 1,
                                   "status": comparison["status"], "reason": "; ".join(comparison["errors"]),
                                   "log": "logs/reference_reproduction.json"})
    statuses = {check["status"] for check in checks}
    report["status"] = "FAIL" if "FAIL" in statuses else ("BLOCKED" if statuses & {"BLOCKED", "SKIP"} else "PASS")
    if not ready:
        report["unexecuted"] = "Dependent fresh-tree checks could not proceed; see the first failure or blocker."
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--reference-run", type=Path, help="Completed parent run to reproduce, excluding timing values")
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_clean(ROOT, args.data, args.work_root, args.source_zip, args.reference_run)
    print(json.dumps({"status": report["status"], "report": str(args.work_root / "clean_validation.json")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
