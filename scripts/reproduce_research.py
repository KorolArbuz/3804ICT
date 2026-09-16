"""Restore an immutable snapshot separately and execute its historical unit suite."""

import argparse
import json
from pathlib import Path
import shutil
import sys

try:
    from .verify_archive import ROOT, safe_relative, verify_snapshot
    from .verify_clean_checkout import record_command
except ImportError:
    from verify_archive import ROOT, safe_relative, verify_snapshot
    from verify_clean_checkout import record_command


def reproduce(snapshot: Path, work_root: Path, python: str = sys.executable) -> dict:
    snapshot, work_root = snapshot.resolve(), work_root.resolve()
    if work_root.exists() or work_root == ROOT or work_root.is_relative_to(ROOT / "archive"):
        raise ValueError("Use a new disposable directory outside the immutable archive")
    verified = verify_snapshot(snapshot)
    if verified["status"] != "PASS":
        raise ValueError(f"Snapshot failed verification: {verified['errors']}")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    tree, logs = work_root / "tree", work_root / "logs"
    tree.mkdir(parents=True)
    logs.mkdir()
    for entry in manifest["files"]:
        destination = tree / safe_relative(entry["old_relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / safe_relative(entry["archived_relative_path"]), destination)
        if entry["category"] == "required-evidence":
            destination.chmod(0o444)
    check = record_command("historical_unit_suite", [python, "-m", "unittest", "discover", "-s", "tests"], tree, logs)
    verified_after = verify_snapshot(snapshot)
    report = {"scope": "Archived historical unit tests; separate from active release validation",
              "status": check["status"] if verified_after["status"] == "PASS" else "FAIL",
              "snapshot_before": verified, "snapshot_after": verified_after,
              "checks": [check], "environment": "Explicit interpreter with installed historical pinned dependencies",
              "limitations": "No expensive archived search, legacy benchmark, raw-data experiment, or native build is executed by this helper."}
    (work_root / "historical_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    snapshots = sorted((ROOT / "archive/research").glob("*/manifest.json"))
    snapshot = args.snapshot or (snapshots[-1].parent if snapshots else None)
    if snapshot is None:
        parser.error("No research snapshot found")
    report = reproduce(snapshot, args.work_root, args.python)
    print(json.dumps({"status": report["status"], "report": str(args.work_root / "historical_validation.json")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
