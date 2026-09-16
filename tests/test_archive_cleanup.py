import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import finalize_archive_move as cleanup


class ArchiveCleanupTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "src/model_quality/old.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"original source\n")
        snapshot = root / "archive/research/example"
        saved = snapshot / "snapshot/src/model_quality/old.py"
        saved.parent.mkdir(parents=True)
        saved.write_bytes(source.read_bytes())
        manifest = {
            "files": [
                {
                    "old_relative_path": "src/model_quality/old.py",
                    "archived_relative_path": "snapshot/src/model_quality/old.py",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "size": source.stat().st_size,
                }
            ]
        }
        (snapshot / "manifest.json").write_text(json.dumps(manifest))
        (root / "docs").mkdir()
        return source, snapshot, saved

    def test_changed_source_stops_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, snapshot, saved = self.fixture(root)
            source.write_bytes(b"new user edits\n")
            with patch.object(cleanup, "ROOT", root), patch.object(
                cleanup, "SNAPSHOT", snapshot
            ), patch.object(
                cleanup,
                "verify_snapshot",
                return_value={"status": "PASS", "manifest_sha256": "fixture"},
            ):
                with self.assertRaisesRegex(ValueError, "Changed source"):
                    cleanup.main(["--apply"])
            self.assertEqual(source.read_bytes(), b"new user edits\n")
            self.assertEqual(saved.read_bytes(), b"original source\n")

    def test_only_verified_copy_removed_unknown_file_retained_and_evidence_idempotent(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, snapshot, saved = self.fixture(root)
            unknown = source.parent / "notes.txt"
            unknown.write_text("Keep user notes")
            with patch.object(cleanup, "ROOT", root), patch.object(
                cleanup, "SNAPSHOT", snapshot
            ), patch.object(
                cleanup,
                "verify_snapshot",
                return_value={"status": "PASS", "manifest_sha256": "fixture"},
            ):
                cleanup.main(["--apply"])
                first = (root / "docs/active_cleanup_manifest.json").read_bytes()
                cleanup.main(["--apply"])
                self.assertEqual(
                    first, (root / "docs/active_cleanup_manifest.json").read_bytes()
                )
            self.assertFalse(source.exists())
            self.assertEqual(saved.read_bytes(), b"original source\n")
            self.assertEqual(unknown.read_text(), "Keep user notes")


if __name__ == "__main__":
    unittest.main()
