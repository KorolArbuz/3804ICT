"""Small release-control tests; no dataset, build or benchmark is executed here."""
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from src.validation.runner import (SCOPES, Session, completion, finalize_release_manifest,
                                   portable_text, reusable_run, scientific_hashes, stable_output_hashes,
                                   verify_recorded_outputs, write_json)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"src/final_knn/classifier.py": "model", "src/preprocessing/transforms.py": "transform",
                        "src/benchmarking/suite.py": "timing", "tests/test_validation.py": "old-test"}
        self.configs = {"baseline": {"config_hash": "b"}, "final": {"config_hash": "f"}}
        self.manifest = {"status": "complete_with_documented_differences", "sources": dict(self.sources),
                         "benchmark_requested": True, "config_hashes": {"baseline": "b", "final": "f"}}
        self.data = {"dataset": {"sha256": "dataset"}}

    def reusable(self, sources=None, data_hash="dataset"):
        return reusable_run(self.manifest, sources or self.sources, self.configs, data_hash, self.data)

    def test_test_and_report_edits_do_not_repeat_expensive_benchmark(self):
        changed = dict(self.sources, **{"tests/test_validation.py": "new-test", "src/reporting/plots.py": "report"})
        self.assertTrue(self.reusable(changed)[0])
        self.assertEqual(len(scientific_hashes(changed)), 3)
        historical = dict(changed, **{"src/cpp_knn/benchmark.py": "archived", "src/cpp_knn/bridge.py": "active"})
        self.assertNotIn("src/cpp_knn/benchmark.py", scientific_hashes(historical))
        self.assertIn("src/cpp_knn/bridge.py", scientific_hashes(historical))

    def test_model_transform_benchmark_and_data_change_prevent_reuse(self):
        for key in ("src/final_knn/classifier.py", "src/preprocessing/transforms.py", "src/benchmarking/suite.py"):
            changed = dict(self.sources)
            changed[key] = "different"
            self.assertFalse(self.reusable(changed)[0])
        self.assertFalse(self.reusable(data_hash="other")[0])
        self.configs["final"]["config_hash"] = "changed-config"
        self.assertFalse(self.reusable()[0])

    def test_config_metadata_only_edits_do_not_invalidate_model_evidence(self):
        self.sources["configs/final.json"] = "original-metadata"
        self.manifest["sources"]["configs/final.json"] = "original-metadata"
        changed = dict(self.sources)
        changed["configs/final.json"] = "added-description"
        self.assertTrue(self.reusable(changed)[0])
        changed["configs/new_model.json"] = "new-unchecked-configuration"
        self.assertFalse(self.reusable(changed)[0])

    def release_fixture(self, folder):
        root = Path(folder)
        root.joinpath("predictions").mkdir()
        prediction = b"score_class_1,y_pred\n0.25,0\n"
        root.joinpath("predictions/final_python.csv").write_bytes(prediction)
        root.joinpath("benchmark_raw.csv").write_bytes(b"seconds\n1.25\n")
        model = {"k": 101, "threshold": 0.3315411365543412}
        digest = hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        config = {"schema_version": 1, "model_group": "final", "model": model, "config_hash": digest}
        original = {"final": config}
        write_json(root / "resolved_configs.json", original)
        manifest = {"run_id": "test-run", "status": "complete", "sources": {"src/final_knn/classifier.py": "frozen-source"},
                    "config_hashes": {"final": digest}}
        write_json(root / "run_manifest.json", manifest)
        evidence = {"run_id": "test-run", "prediction_sha256": {"final_python": hashlib.sha256(prediction).hexdigest()}}
        return root, original, manifest, evidence

    def test_release_finalization_preserves_inference_bytes_and_avoids_hash_cycles(self):
        with tempfile.TemporaryDirectory() as folder:
            root, original, manifest, evidence = self.release_fixture(folder)
            original_bytes = (root / "resolved_configs.json").read_bytes()
            timing_bytes = (root / "benchmark_raw.csv").read_bytes()
            current = json.loads(json.dumps(original))
            current["final"]["model_id"] = "frozen-v2"
            current["final"]["threshold_source"] = {"path": "archive/research/frozen.json", "json_pointer": "/min_fp_recall_50/threshold"}
            sources = {**manifest["sources"], "configs/final.json": "metadata-supplement", "tests/test_validation.py": "release-tests"}
            write_json(root / "validation.json", {"status": "PASS"})
            write_json(root / "report_manifest.json", {"source": "saved outputs"})
            finalized = finalize_release_manifest(root, sources, current, evidence)
            updated = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(updated["sources"], manifest["sources"])
            self.assertEqual(updated["release_sources"], sources)
            self.assertEqual((root / "resolved_configs.json").read_bytes(), original_bytes)
            self.assertEqual((root / "benchmark_raw.csv").read_bytes(), timing_bytes)
            self.assertEqual(finalized["metadata_only_configuration_changes"]["final"], ["model_id", "threshold_source"])
            self.assertNotIn("run_manifest.json", updated["outputs"])
            self.assertNotIn("validation.json", updated["outputs"])
            self.assertNotIn("report_manifest.json", updated["outputs"])
            self.assertEqual(updated["outputs"], stable_output_hashes(root))
            verify_recorded_outputs(root, updated)
            (root / "benchmark_raw.csv").write_bytes(b"seconds\n0.125\n")
            with self.assertRaises(ValueError):
                verify_recorded_outputs(root, updated)

    def test_release_finalization_rejects_changed_model_or_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root, configs, manifest, evidence = self.release_fixture(folder)
            changed = json.loads(json.dumps(configs))
            changed["final"]["model"]["threshold"] = 0.5
            with self.assertRaises(ValueError):
                finalize_release_manifest(root, manifest["sources"], changed, evidence)
            (root / "predictions/final_python.csv").write_bytes(b"changed predictions")
            with self.assertRaises(ValueError):
                finalize_release_manifest(root, manifest["sources"], configs, evidence)

    def test_partial_or_missing_benchmark_cannot_be_reused(self):
        self.manifest["status"] = "partial"
        self.assertFalse(self.reusable()[0])
        self.manifest["status"] = "complete"
        self.manifest["benchmark_requested"] = False
        self.assertFalse(self.reusable()[0])

    def test_completion_never_hides_fail_block_skip(self):
        checks = [{"scope": scope, "status": "PASS"} for scope in SCOPES]
        self.assertEqual(completion(checks, full=True, differences=True)[0], "complete_with_documented_differences")
        self.assertEqual(completion(checks, full=False)[0], "partial")
        for bad in ("FAIL", "SKIP", "BLOCKED"):
            self.assertEqual(completion(checks + [{"scope": SCOPES[0], "status": bad}], full=True)[0], "partial")

    def test_portable_commands_hide_user_paths(self):
        windows = "C:" + "/Users/" + "someone/project/log.txt"
        unix = "/" + "home/someone/project/log.txt"
        self.assertEqual(portable_text(windows, root=Path("unrelated")), "{user}/project/log.txt")
        self.assertEqual(portable_text(unix, root=Path("unrelated")), "{user}/project/log.txt")

    def test_nonzero_command_and_missing_tool_have_explicit_results(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Session(Path(folder) / "session")
            with patch("src.validation.runner.subprocess.Popen", side_effect=FileNotFoundError("missing tool")):
                record = session.command("missing", "build correctness", ["nonexistent"])
            self.assertEqual(record["status"], "BLOCKED")
            self.assertIn("command", record)
            self.assertIn("started", record)
            self.assertIn("finished", record)
            self.assertTrue((session.directory / "checks.json").is_file())


if __name__ == "__main__":
    unittest.main()
