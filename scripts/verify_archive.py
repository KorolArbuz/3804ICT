"""Check the immutable research snapshot against its raw-byte manifest."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RETENTION_OVERLAY_PATH = "snapshot/results/.gitignore"
RETENTION_OVERLAY_BYTES = (
    "# Archive retention overlay added during finalization; not an original snapshot file.\n"
    "# Re-include preserved evidence without editing the historical parent .gitignore.\n"
    "!**\n"
).encode("utf-8")


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*path.parts)


def verify_snapshot(snapshot: Path) -> dict:
    snapshot = snapshot.resolve()
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors, seen = [], set()
    byte_count = 0
    for entry in manifest["files"]:
        relative = entry["archived_relative_path"]
        try:
            destination = snapshot / safe_relative(relative)
            original = safe_relative(entry["old_relative_path"])
            if relative != (Path("snapshot") / original).as_posix():
                raise ValueError("Archived path does not preserve original path")
            if relative in seen:
                raise ValueError("Duplicate manifest entry")
            seen.add(relative)
            if destination.is_symlink() or not destination.resolve().is_relative_to(snapshot):
                raise ValueError("Snapshot entry leaves the snapshot tree")
            content = destination.read_bytes()
            if len(content) != entry["size"]:
                raise ValueError("Byte size differs")
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("SHA-256 differs")
            byte_count += len(content)
        except (OSError, ValueError) as error:
            errors.append({"path": relative, "reason": str(error)})
    overlays = []
    originals = {entry["old_relative_path"] for entry in manifest["files"]}
    needs_retention = ".gitignore" in originals and any(name.startswith("results/") for name in originals)
    if needs_retention and RETENTION_OVERLAY_PATH not in seen:
        path = snapshot / RETENTION_OVERLAY_PATH
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(snapshot):
                raise ValueError("Retention overlay leaves snapshot tree")
            content = path.read_bytes()
            if content != RETENTION_OVERLAY_BYTES:
                raise ValueError("Retention overlay differs from its declared nonhistorical content")
            seen.add(RETENTION_OVERLAY_PATH)
            overlays.append({"path": RETENTION_OVERLAY_PATH, "size": len(content),
                             "sha256": hashlib.sha256(content).hexdigest(),
                             "role": "New Git retention metadata; excluded from original manifest file count"})
        except (OSError, ValueError) as error:
            errors.append({"path": RETENTION_OVERLAY_PATH, "reason": str(error)})
    actual = {p.relative_to(snapshot).as_posix() for p in (snapshot / "snapshot").rglob("*")
              if p.is_file()}
    for relative in sorted(actual - seen):
        errors.append({"path": relative, "reason": "Unlisted file in immutable snapshot"})
    return {"snapshot_id": manifest["snapshot_id"], "status": "FAIL" if errors else "PASS",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "files": len(manifest["files"]), "verified_bytes": byte_count,
            "retention_overlays": overlays, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=Path("verification/archive_validation.json"))
    args = parser.parse_args(argv)
    snapshots = [args.snapshot] if args.snapshot else sorted((ROOT / "archive/research").glob("*/manifest.json"))
    if not args.snapshot:
        snapshots = [p.parent for p in snapshots]
    reports = [verify_snapshot(p) for p in snapshots]
    report = {"status": "PASS" if reports and all(r["status"] == "PASS" for r in reports) else "FAIL",
              "snapshots": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
