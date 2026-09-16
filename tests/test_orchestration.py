import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.common.experiment import load_configs
from src.common.utils import write_json
from src.experiment import parser, run, _raw_predictions
from src.preprocessing.prepared import prepare
from tests.test_prepared_data import sample_frame


class OrchestrationTests(unittest.TestCase):
    def test_partial_run_is_explicit_and_each_run_has_new_identity(self):
        with tempfile.TemporaryDirectory(prefix="knn paths with spaces ") as temporary:
            root = Path(temporary)
            data = root / "sample data.csv"
            frame = sample_frame()
            frame["target"] = np.arange(len(frame)) % 2
            frame.to_csv(data, index=False)
            args = parser().parse_args(
                [
                    "--data",
                    str(data),
                    "--only",
                    "baseline_python_v5_1",
                    "final_sklearn",
                    "--output-root",
                    str(root / "runs"),
                ]
            )
            # Plotting is independently covered; this checks real data/model/evaluator orchestration.
            with patch("src.reporting.plots.generate_report"):
                status, first = run(args)
                next_status, second = run(args)
            self.assertEqual((status, next_status), (1, 1))
            self.assertNotEqual(first, second)
            manifest = json.loads((first / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(len(manifest["completed_implementations"]), 2)
            self.assertFalse((first / "predictions/final_weka.csv").exists())

    def test_missing_dataset_compact_failure_evidence_and_no_old_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = parser().parse_args(
                [
                    "--data",
                    str(root / "missing.csv"),
                    "--output-root",
                    str(root / "missing_data"),
                ]
            )
            status, directory = run(args)
            self.assertEqual(status, 1)
            manifest = json.loads((directory / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(list(directory.rglob("*.png")))
            self.assertFalse(list(directory.rglob("*.csv")))

    def test_external_output_rejects_reordered_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.csv"
            path.write_text("test_position,score_class_1,y_pred\n1,0.3,0\n0,0.7,1\n")
            with self.assertRaises(ValueError):
                _raw_predictions(path, 2)


if __name__ == "__main__":
    unittest.main()
