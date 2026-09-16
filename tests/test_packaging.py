import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.export_submission import (REQUIRED_IMPLEMENTATIONS, appendix_bytes, export_submission,
                                       source_files, verify_zip, write_zip)
from scripts.verify_archive import RETENTION_OVERLAY_BYTES, safe_relative, verify_snapshot
from scripts.verify_clean_checkout import compare_reproduction, prepare_tree


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def fixture_run(self):
        prefix = "results/final/current/"
        self.write(prefix + "run_manifest.json", '{"status":"complete"}')
        for name in REQUIRED_IMPLEMENTATIONS:
            self.write(prefix + f"predictions/{name}.csv", "row_id,y_true,score,y_pred\n1,1,0.7,1\n")
        for name in ("metrics.csv", "benchmark_raw.csv", "benchmark_summary.csv", "agreement.csv"):
            self.write(prefix + name, "sample\n1\n")
        self.write(prefix + "benchmark_raw.csv", "implementation_id,phase,is_warmup,seconds\n" +
                   "".join(f"{name},prediction,False,1\n" for name in sorted(REQUIRED_IMPLEMENTATIONS)))
        self.write(prefix + "validation.json", '{"status":"PASS"}')
        return Path(prefix)

    def test_current_edits_and_determinism_and_allowlist(self):
        source = self.write("src/active.py", "value = 'uncommitted edit'\n")
        self.write("src/__pycache__/active.pyc", "excluded")
        self.write("src/credentials/.env", "excluded")
        self.write("weka/target/third-party.jar", "excluded")
        self.write("weka/src/main/java/Runner.java", "class Runner {}\n")
        self.write("data/raw/UCI_Credit_Card.csv", "excluded raw data")
        self.write("data/raw/README.md", "Dataset placement")
        self.write("archive/README.md", "History supplied separately")
        self.write("archive/research/old/snapshot/src/old.py", "excluded history")
        self.write("build/knn.exe", "excluded executable")
        self.write("dist/old.zip", "excluded export")
        self.write(".git/config", "excluded metadata")
        self.write(".venv/pyvenv.cfg", "excluded environment")
        run = self.fixture_run()
        self.write(run / "inputs/train.csv", "excluded matrix")
        self.write(run / "release_validation_external.json", '{"proof":"previous ZIP execution must remain external"}')
        self.write("results/final/other/metrics.csv", "excluded other run")
        first, second = self.root / "dist/first.zip", self.root / "dist/second.zip"
        export_submission(self.root, first, run)
        export_submission(self.root, second, run)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(archive.read("src/active.py"), source.read_bytes())
            self.assertIn("weka/src/main/java/Runner.java", archive.namelist())
            self.assertIn("appendix_source_code.md", archive.namelist())
            self.assertFalse(any(name.startswith(("build/", "dist/", ".git/", ".venv/", "archive/research/")) for name in archive.namelist()))
            self.assertNotIn("data/raw/UCI_Credit_Card.csv", archive.namelist())
            self.assertNotIn("results/final/current/inputs/train.csv", archive.namelist())
            self.assertNotIn("results/final/current/release_validation_external.json", archive.namelist())
        self.assertEqual(verify_zip(first)["status"], "PASS")

    def test_incomplete_run_and_missing_implementation_are_rejected(self):
        run = self.fixture_run()
        self.write(run / "run_manifest.json", '{"status":"partial"}')
        with self.assertRaisesRegex(ValueError, "completed"):
            export_submission(self.root, self.root / "submission.zip", run)
        self.write(run / "run_manifest.json", '{"status":"complete"}')
        (self.root / run / "predictions/final_weka.csv").unlink()
        with self.assertRaisesRegex(ValueError, "final_weka"):
            export_submission(self.root, self.root / "submission.zip", run)

    def test_local_paths_are_rejected_without_rewriting_evidence(self):
        run = self.fixture_run()
        metadata = self.write(run / "environment.json", json.dumps({"python": "C:" + "/Users/" + "student/python.exe"}))
        original = metadata.read_bytes()
        with self.assertRaisesRegex(ValueError, "Personal absolute paths"):
            export_submission(self.root, self.root / "submission.zip", run)
        self.assertEqual(metadata.read_bytes(), original)

    def test_appendix_hashes_active_source_once(self):
        path = self.write("src/module.py", "print('source')\n")
        self.write("archive/research/snapshot/src/module.py", "old source")
        appendix = appendix_bytes(source_files(self.root)).decode()
        self.assertEqual(appendix.count("## src/module.py"), 1)
        self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), appendix)
        self.assertNotIn("old source", appendix)

    def test_source_allowlist_omits_archived_research_before_cleanup(self):
        for name in ("src/model_quality/search.py", "src/model_quality_v2/operating_point.py",
                     "src/tuning/search.py", "tests/test_model_quality.py",
                     "tests/test_model_quality_threshold.py", "tests/test_model_quality_v2.py",
                     "tests/test_model_quality_v2_operating_point.py", "src/cpp_knn/benchmark.py",
                     "src/cpp_knn/evidence.py", "src/cpp_knn/export_data.py", "src/cpp_knn/verify.py",
                     "src/benchmarking/benchmark_sessions.py", "src/benchmarking/generate_figures.py"):
            self.write(name, "historical_only = True\n")
        expected = {"src/final_knn/classifier.py", "src/custom_knn/classifier.py",
                    "src/benchmarking/suite.py", "src/benchmarking/evidence.py",
                    "src/benchmarking/prepared_child.py", "src/cpp_knn/bridge.py",
                    "tests/test_final_contract.py", ".gitattributes"}
        for name in expected:
            self.write(name, "active = True\n")
        self.assertEqual(set(source_files(self.root)), expected)

    def test_zip_tampering_and_traversal_rejected(self):
        output = self.root / "out.zip"
        write_zip(output, {"src/file.py": b"a=1\n"}, "final-submission")
        with zipfile.ZipFile(output, "a") as archive:
            archive.writestr("unexpected.txt", "extra")
        with self.assertRaisesRegex(ValueError, "absent"):
            verify_zip(output)
        for value in ("../escape", "/absolute", "C:/absolute", "path\\escape"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative(value)

    def test_clean_tree_excludes_old_products(self):
        self.write("src/active.py", "pass\n")
        self.write("build/old.exe", "old")
        self.write("data/processed/train.csv", "old")
        self.write("results/final/old/predictions.csv", "old")
        self.write(".venv/pyvenv.cfg", "old")
        tree = self.root / "verification/clean/tree"
        report = prepare_tree(self.root, tree)
        self.assertEqual(report["mode"], "current working tree copy")
        self.assertEqual(sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file()), ["src/active.py"])
        with self.assertRaises(FileExistsError):
            prepare_tree(self.root, tree)

    def test_archive_detects_changed_missing_and_extra_bytes(self):
        payload = self.write("snapshot/src/a.py", "unchanged\r\n")
        manifest = {"snapshot_id": "test", "files": [{"old_relative_path": "src/a.py",
                    "archived_relative_path": "snapshot/src/a.py", "size": payload.stat().st_size,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}]}
        self.write("manifest.json", json.dumps(manifest))
        self.assertEqual(verify_snapshot(self.root)["status"], "PASS")
        payload.write_bytes(b"changed")
        self.assertEqual(verify_snapshot(self.root)["status"], "FAIL")
        payload.unlink()
        self.assertEqual(verify_snapshot(self.root)["status"], "FAIL")
        self.write("snapshot/unlisted.txt", "new")
        self.assertTrue(any("Unlisted" in row["reason"] for row in verify_snapshot(self.root)["errors"]))

    def test_retention_overlay_is_separate_from_original_manifest(self):
        paths = [self.write("snapshot/.gitignore", "results/predictions_*.csv\n"),
                 self.write("snapshot/results/predictions_custom.csv", "row,prediction\n0,1\n")]
        entries = [{"old_relative_path": path.relative_to(self.root / "snapshot").as_posix(),
                    "archived_relative_path": path.relative_to(self.root).as_posix(),
                    "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in paths]
        manifest_path = self.write("manifest.json", json.dumps({"snapshot_id": "fixture", "files": entries}))
        manifest_bytes = manifest_path.read_bytes()
        self.assertEqual(verify_snapshot(self.root)["status"], "FAIL")
        overlay = self.root / "snapshot/results/.gitignore"
        overlay.write_bytes(RETENTION_OVERLAY_BYTES)
        report = verify_snapshot(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["files"], 2)
        self.assertEqual(len(report["retention_overlays"]), 1)
        self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
        overlay.write_bytes(b"!**\n# unexpected alteration\n")
        self.assertEqual(verify_snapshot(self.root)["status"], "FAIL")

    def fixture_reproduction(self):
        data = {"dataset": {"sha256": "same-data"}, "split": {"test_row_ids_hash": "same-ids", "y_test_hash": "same-y"},
                "groups": {group: {"config_hash": group, "training_shape": [2, 2], "test_shape": [2, 2],
                            "dtype": "<f8", "feature_names": ["a", "b"], "train_numeric_hash": "train",
                            "test_numeric_hash": "test"} for group in ("baseline", "final")}}
        for role in ("reference", "child"):
            self.write(f"{role}/run_manifest.json", json.dumps({"run_id": role, "status": "complete",
                       "config_hashes": {group: group for group in ("baseline", "final")}}))
            self.write(f"{role}/data_manifest.json", json.dumps(data))
            for implementation in sorted(REQUIRED_IMPLEMENTATIONS):
                group = "baseline" if implementation.startswith("baseline_") else "final"
                self.write(f"{role}/predictions/{implementation}.csv",
                    "run_id,implementation_id,model_group,config_hash,test_position,row_id,original_id,y_true,score_class_1,y_pred,decision_rule,threshold\n" +
                    "".join(f"{role},{implementation},{group},{group},{i},{i},client{i},{i},{score},{i},inclusive,0.3315411365543412\n"
                            for i, score in enumerate((.1, .9))))
        return self.root / "reference", self.root / "child"

    def change_prediction(self, child, implementation, column, value):
        path = child / "predictions" / f"{implementation}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        rows[-1][column] = value
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_clean_reference_all_models_and_tolerance(self):
        reference, child = self.fixture_reproduction()
        self.change_prediction(child, "final_cpp", "score_class_1", "0.9000000000005")
        report = compare_reproduction(reference, child)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(set(report["implementations"]), REQUIRED_IMPLEMENTATIONS)
        self.assertEqual(report["implementations"]["final_cpp"]["exact_label_matches"], 2)
        self.assertGreater(report["implementations"]["final_cpp"]["max_absolute_score_difference"], 0)
        self.change_prediction(child, "final_cpp", "score_class_1", "0.90001")
        self.assertEqual(compare_reproduction(reference, child)["status"], "FAIL")

    def test_clean_reference_detects_identity_labels_and_feature_order(self):
        reference, child = self.fixture_reproduction()
        self.change_prediction(child, "final_python", "row_id", "unexpected-id")
        self.change_prediction(child, "final_weka", "y_pred", "0")
        data_path = child / "data_manifest.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data["groups"]["final"]["feature_names"].reverse()
        data_path.write_text(json.dumps(data), encoding="utf-8")
        report = compare_reproduction(reference, child)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["implementations"]["final_python"]["identity_mismatches"], 1)
        self.assertEqual(report["implementations"]["final_weka"]["label_disagreements"], 1)
        self.assertEqual(report["matrices"]["final"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
