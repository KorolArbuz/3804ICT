"""Remove only verified, superseded research copies after release checks.

The protected snapshot is never edited. Unknown files and changed copies stop this
operation, rather than being removed under a broad directory deletion.
"""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from .verify_archive import verify_snapshot
except ImportError:
    from verify_archive import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "archive/research/pre_final_df1a2c3f67e4"
RESEARCH_PREFIXES = ("src/model_quality/", "src/model_quality_v2/", "src/tuning/")
CPP_HELPERS = {
    "src/cpp_knn/benchmark.py",
    "src/cpp_knn/evidence.py",
    "src/cpp_knn/export_data.py",
    "src/cpp_knn/verify.py",
}
BENCHMARK_KEEP = {"__init__.py", "__main__.py", "README.md"}


def is_superseded(relative):
    return (
        relative.startswith(RESEARCH_PREFIXES)
        or relative.startswith("results/")
        or relative in CPP_HELPERS
        or relative == ".vscode/tasks.json"
        or (
            relative.startswith("tests/test_model_quality") and relative.endswith(".py")
        )
        or (
            relative.startswith("src/benchmarking/")
            and Path(relative).name not in BENCHMARK_KEEP
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the individually verified superseded copies",
    )
    args = parser.parse_args(argv)
    report = verify_snapshot(SNAPSHOT)
    if report["status"] != "PASS":
        raise RuntimeError("Archive verification failed; no copies will be removed")
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    candidates = []
    for entry in manifest["files"]:
        relative = entry["old_relative_path"]
        if not is_superseded(relative):
            continue
        source = ROOT / relative
        if not source.exists():
            continue
        resolved = source.resolve()
        if (
            not resolved.is_relative_to(ROOT)
            or source.is_symlink()
            or not source.is_file()
        ):
            raise ValueError(f"Unsafe cleanup target: {relative}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != entry["sha256"] or source.stat().st_size != entry["size"]:
            raise ValueError(f"Changed source is not eligible for cleanup: {relative}")
        candidates.append(
            {
                "path": relative,
                "sha256": digest,
                "archived_path": (
                    SNAPSHOT.relative_to(ROOT) / entry["archived_relative_path"]
                ).as_posix(),
            }
        )
    if args.apply:
        output = ROOT / "docs/active_cleanup_manifest.json"
        previous = (
            json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
        )
        if (
            previous is not None
            and previous["archive_manifest_sha256"] != report["manifest_sha256"]
        ):
            raise ValueError("Existing cleanup evidence refers to a different archive")
        for entry in candidates:
            source = ROOT / entry["path"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Source changed during cleanup: {entry['path']}")
            source.unlink()
        # Only empty parent directories are removed; caches/unknown contents are retained.
        parents = {
            parent
            for entry in candidates
            for parent in (ROOT / entry["path"]).parents
            if parent != ROOT and parent.is_relative_to(ROOT)
        }
        for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        if verify_snapshot(SNAPSHOT)["status"] != "PASS":
            raise RuntimeError("Unexpected archive change after cleanup")
        retained_evidence = {
            entry["path"]: entry
            for entry in (previous or {}).get("deleted_active_copies", [])
        }
        retained_evidence.update({entry["path"]: entry for entry in candidates})
        output.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "archive_manifest_sha256": report["manifest_sha256"],
                    "deleted_active_copies": [
                        retained_evidence[key] for key in sorted(retained_evidence)
                    ],
                    "preserved": "All original archive bytes, baseline classifier/constants, raw data and unknown files",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "mode": "applied" if args.apply else "preview",
                "verified_candidates": len(candidates),
                "paths": [item["path"] for item in candidates],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
