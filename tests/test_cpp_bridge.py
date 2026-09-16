"""Dataset-free CLI contract checks when the current native build is available."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from src.cpp_knn import bridge


class CppBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.executable = (
                Path(os.environ["CPP_KNN_TEST_EXECUTABLE"])
                if os.environ.get("CPP_KNN_TEST_EXECUTABLE")
                else bridge.executable_path()
            )
        except RuntimeError as error:
            raise unittest.SkipTest(
                "Build C++ via CMake for CLI integration tests"
            ) from error
        # A stale binary is a test failure, rather than a skip.
        cls.identity = bridge.validate_executable(cls.executable)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        (self.path / "train_features.csv").write_text("-1\n1\n", encoding="utf-8")
        (self.path / "train_labels.csv").write_text("0\n1\n", encoding="utf-8")
        (self.path / "test_features.csv").write_text("0\n1\n", encoding="utf-8")
        self.config = {
            "k": 2,
            "weights": "distance",
            "threshold": 0.5,
            "config_hash": "synthetic-contract",
        }

    def tearDown(self):
        self.directory.cleanup()

    def test_score_and_label_materialization(self):
        result = bridge.score(
            self.executable, self.path, self.path / "output", self.config
        )
        self.assertEqual(result["scores_class_1"], [0.5, 1.0])
        self.assertEqual(result["predicted_labels"], [1, 1])
        self.assertEqual(result["config_id"], self.config["config_hash"])
        self.assertEqual(result["source_id"], bridge.source_id())

    def test_persistent_warmup_and_repeated_fresh_predictions(self):
        with bridge.PersistentCpp(
            self.executable, self.path, self.config, warmups=3
        ) as worker:
            self.assertEqual(len(worker.ready["warmup_seconds"]), 3)
            first, second = worker.predict(), worker.predict()
            self.assertEqual(first["checksum"], second["checksum"])
            self.assertEqual(first["prediction_count"], 2)
            self.assertGreaterEqual(first["seconds"], 0)
        self.assertEqual(worker.process.returncode, 0)

    def test_cli_rejects_invalid_options_and_nonfinite_input(self):
        base = bridge.command_arguments(self.executable, self.path, self.config)
        result = subprocess.run(
            base + ["--unknown", "x", "--output", str(self.path / "bad.json")],
            cwd=bridge.ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.path / "bad.json").exists())
        (self.path / "test_features.csv").write_text("nan\n", encoding="utf-8")
        result = subprocess.run(
            base + ["--output", str(self.path / "bad.json")],
            cwd=bridge.ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.path / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
