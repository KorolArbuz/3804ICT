"""Deterministic allowlist export of the current working files, including local edits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

try:
    from .verify_archive import safe_relative, verify_snapshot
except ImportError:
    from verify_archive import safe_relative, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".cpp", ".hpp", ".h", ".cc", ".cxx", ".java"}
TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".xml", ".toml", ".yaml", ".yml"}
EVIDENCE_SUFFIXES = {".md", ".txt", ".json", ".csv", ".png", ".svg"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist",
                  ".pytest_cache", ".mypy_cache", ".ruff_cache", "toolchain", "credentials"}
ROOT_FILES = {"README.md", "run_all.py", "CMakeLists.txt", "requirements.txt", ".gitignore", ".gitattributes",
              "LICENSE", "LICENSE.md", "pyproject.toml", "CMakePresets.json"}
REQUIRED_IMPLEMENTATIONS = {"baseline_python_v5_1", "final_python", "final_sklearn", "final_cpp", "final_weka"}
# These copies remain in the working tree until the first clean validation is
# complete. Their byte-verified originals are delivered in the research ZIP.
RESEARCH_PREFIXES = ("src/model_quality/", "src/model_quality_v2/", "src/tuning/")
RESEARCH_TESTS = {"tests/test_model_quality.py", "tests/test_model_quality_threshold.py",
                  "tests/test_model_quality_v2.py", "tests/test_model_quality_v2_operating_point.py"}
ARCHIVED_CPP_HELPERS = {"src/cpp_knn/benchmark.py", "src/cpp_knn/evidence.py",
                        "src/cpp_knn/export_data.py", "src/cpp_knn/verify.py"}
ACTIVE_BENCHMARK_FILES = {"__init__.py", "__main__.py", "README.md", "suite.py", "prepared_child.py", "evidence.py"}


def active_source_path(relative: str) -> bool:
    if relative.startswith(RESEARCH_PREFIXES) or relative in RESEARCH_TESTS | ARCHIVED_CPP_HELPERS:
        return False
    return not relative.startswith("src/benchmarking/") or Path(relative).name in ACTIVE_BENCHMARK_FILES


def reject_personal_paths(contents: dict[str, bytes]) -> None:
    """Reject leaked local paths instead of changing provenance-bearing bytes."""
    pattern = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+|/(?:home|Users)/[^/\s\"']+)")
    offending = []
    for name, content in contents.items():
        if Path(name).suffix in SOURCE_SUFFIXES | TEXT_SUFFIXES | {".svg"}:
            if pattern.search(content.decode("utf-8-sig")):
                offending.append(name)
    if offending:
        raise ValueError("Personal absolute paths must be removed from portable metadata before export: " + ", ".join(sorted(offending)))


def allowed_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (not any(part in EXCLUDED_PARTS or part.startswith(".venv") for part in relative.parts)
            and not path.name.startswith(".env") and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()))


def source_files(root: Path) -> dict[str, Path]:
    """Use current bytes, never Git HEAD; no generated predictions or raw data."""
    root = root.resolve()
    files = {name: root / name for name in ROOT_FILES if (root / name).is_file() and allowed_file(root / name, root)}
    directories = {"src": SOURCE_SUFFIXES | {".md"}, "tests": SOURCE_SUFFIXES | TEXT_SUFFIXES,
                   "scripts": {".py", ".md"}, "configs": {".json"}, "docs": {".md", ".json", ".png", ".svg"},
                   "weka/src": {".java", ".xml", ".properties", ".csv", ".arff"},
                   "report_assets": EVIDENCE_SUFFIXES}
    for directory, suffixes in directories.items():
        for path in (root / directory).rglob("*"):
            if (path.is_file() and path.suffix in suffixes and allowed_file(path, root)
                    and path.name != "appendix_source_code.md"
                    and active_source_path(path.relative_to(root).as_posix())):
                files[path.relative_to(root).as_posix()] = path
    for relative in ("weka/pom.xml", "data/raw/README.md", "archive/README.md"):
        if (root / relative).is_file() and allowed_file(root / relative, root):
            files[relative] = root / relative
    return dict(sorted(files.items()))


def select_run(root: Path, run_dir: Path | None = None) -> Path:
    final_root = (root / "results/final").resolve()
    if run_dir is not None:
        candidate = run_dir if run_dir.is_absolute() else root / run_dir
        candidates = [candidate.resolve()]
    else:
        candidates = sorted((p.parent for p in final_root.glob("*/run_manifest.json")), reverse=True)
    for candidate in candidates:
        if not candidate.is_relative_to(final_root) or candidate == final_root:
            raise ValueError("Submission run must be inside results/final/<run_id>")
        manifest = json.loads((candidate / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") in {"completed", "complete", "complete_with_documented_differences"}:
            return candidate
        if run_dir is not None:
            raise ValueError("Submission requires a successfully completed five-model final run")
    raise ValueError("No completed final run found; execute run_all.py first")


def run_files(root: Path, run_dir: Path) -> dict[str, Path]:
    files = {}
    for path in run_dir.rglob("*"):
        relative_run = path.relative_to(run_dir)
        if (path.is_file() and path.suffix in EVIDENCE_SUFFIXES and allowed_file(path, root)
                and relative_run.as_posix() != "release_validation_external.json"
                and not any(part in {"data", "inputs", "prepared", "matrices", "canonical", "native"} for part in relative_run.parts)):
            files[path.relative_to(root).as_posix()] = path
    predictions = {name for name in REQUIRED_IMPLEMENTATIONS
                   if any(name in path.name and path.suffix == ".csv" and ("prediction" in path.name or "predictions" in path.parts)
                          for path in files.values())}
    if predictions != REQUIRED_IMPLEMENTATIONS:
        raise ValueError(f"Completed run is missing prediction CSVs: {sorted(REQUIRED_IMPLEMENTATIONS - predictions)}")
    if not any("metrics" in p.name and p.suffix == ".csv" for p in files.values()):
        raise ValueError("Completed run is missing metrics CSV")
    for required in ("benchmark_raw.csv", "benchmark_summary.csv", "agreement.csv", "validation.json"):
        if not (run_dir / required).is_file():
            raise ValueError(f"Completed run is missing required submission evidence: {required}")
    with (run_dir / "benchmark_raw.csv").open(newline="", encoding="utf-8") as stream:
        measured = {row.get("implementation_id") for row in csv.DictReader(stream)
                    if row.get("phase") == "prediction" and row.get("is_warmup", "").lower() in {"false", "0"}}
    if measured != REQUIRED_IMPLEMENTATIONS:
        raise ValueError("Submission requires measured prediction timings for all five implementations")
    return files


def appendix_bytes(files: dict[str, Path]) -> bytes:
    languages = {".py": "python", ".java": "java", ".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".cc": "cpp", ".cxx": "cpp"}
    lines = ["# Active source code appendix", "", "Generated from the current working files. Research source is supplied separately in the history archive.", ""]
    for relative, path in sorted(files.items()):
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        fence = "````" if "```" in text else "```"
        lines.extend([f"## {relative}", "", f"Language: {languages[path.suffix]}; SHA-256: `{hashlib.sha256(content).hexdigest()}`.",
                      "", fence + languages[path.suffix], text.rstrip("\n"), fence, ""])
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_zip(output: Path, contents: dict[str, bytes], kind: str) -> dict:
    entries = [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
               for name, content in sorted(contents.items())]
    manifest = {"schema_version": 1, "kind": kind, "source": "current working tree bytes",
                "files": entries}
    contents = dict(contents)
    contents["submission_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(contents.items()):
            safe_relative(name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {"status": "PASS", "kind": kind, "files": len(contents), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def verify_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        for info in archive.infolist():
            safe_relative(info.filename)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Symbolic links are forbidden in submission ZIPs")
        manifest = json.loads(archive.read("submission_manifest.json"))
        expected = {"submission_manifest.json"}
        for entry in manifest["files"]:
            if entry["path"] in expected:
                raise ValueError("Duplicate manifest entry")
            expected.add(entry["path"])
            content = archive.read(entry["path"])
            if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP byte verification failed: {entry['path']}")
        if set(names) != expected:
            raise ValueError("ZIP contains files absent from its manifest")
        return {"status": "PASS", "files": len(names), "kind": manifest["kind"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def export_submission(root: Path, output: Path, run_dir: Path | None = None) -> dict:
    root = root.resolve()
    selected = select_run(root, run_dir)
    files = source_files(root)
    files.update(run_files(root, selected))
    contents = {name: path.read_bytes() for name, path in files.items()}
    contents["appendix_source_code.md"] = appendix_bytes(source_files(root))
    reject_personal_paths(contents)
    report = write_zip(output, contents, "final-submission")
    report["run_dir"] = selected.relative_to(root).as_posix()
    report["verification"] = verify_zip(output)
    return report


def export_history(root: Path, output: Path) -> dict:
    manifests = sorted((root / "archive/research").glob("*/manifest.json"))
    if not manifests:
        raise ValueError("No research snapshots found")
    contents = {}
    if (root / "archive/README.md").is_file():
        contents["archive/README.md"] = (root / "archive/README.md").read_bytes()
    for manifest in manifests:
        report = verify_snapshot(manifest.parent)
        if report["status"] != "PASS":
            raise ValueError(f"Research snapshot verification failed: {report['errors']}")
        indexed = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [manifest] + [manifest.parent / name for name in ("README.md", "ARCHIVE_NOTES.md")]
        paths += [manifest.parent / safe_relative(entry["archived_relative_path"]) for entry in indexed["files"]]
        paths += [manifest.parent / safe_relative(entry["path"]) for entry in report["retention_overlays"]]
        for path in paths:
            if path.is_file():
                contents[path.relative_to(root).as_posix()] = path.read_bytes()
    report = write_zip(output, contents, "research-history")
    report["verification"] = verify_zip(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/3804ICT-final-submission.zip"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--history-output", type=Path, default=Path("dist/3804ICT-research-history.zip"))
    args = parser.parse_args(argv)
    report = {"submission": export_submission(ROOT, args.output, args.run_dir),
              "research_history": export_history(ROOT, args.history_output)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
