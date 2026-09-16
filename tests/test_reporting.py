import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common.experiment import IMPLEMENTATIONS, array_hash, load_configs, model_group, sha256
from src.evaluation.final import evaluate, prediction_frame
from src.reporting.plots import load_evidence, plot_results, prediction_timings


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "predictions").mkdir()
        configs = load_configs()
        self.configs = configs
        data = {"y_test": np.array([0, 1, 0, 1], dtype=np.int64), "row_ids": np.arange(4, dtype=np.int64),
                "original_ids": np.array(["a", "b", "c", "d"])}
        data_manifest = {"split": {"test_row_ids_hash": array_hash(data["row_ids"]),
                                  "y_test_hash": array_hash(data["y_test"]),
                                  "original_test_ids_hash": array_hash(data["original_ids"])}}
        for name in IMPLEMENTATIONS:
            config = configs[model_group(name)]
            scores = np.array([.2, .9, .35, .7])
            threshold = config["model"]["threshold"]
            labels = (scores > .5 if threshold is None else scores >= threshold).astype(int)
            frame = prediction_frame("fixture", name, config, data, scores, labels)
            frame.to_csv(self.run / "predictions" / f"{name}.csv", index=False, float_format="%.17g")
        evaluate(self.run, "fixture", IMPLEMENTATIONS, configs, data_manifest)
        self.manifest = {"run_id": "fixture", "config_hashes": {name: config["config_hash"] for name, config in configs.items()}}
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_stale_configuration_rejected_before_report(self):
        self.manifest["config_hashes"]["final"] = "stale"
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration hashes"):
            plot_results(self.run)
        self.assertFalse((self.run / "figures").exists())

    def test_metrics_recomputed_from_saved_predictions(self):
        table = pd.read_csv(self.run / "metrics.csv")
        table.loc[0, "FP"] += 1
        table.to_csv(self.run / "metrics.csv", index=False)
        with self.assertRaisesRegex(ValueError, "metric arithmetic"):
            load_evidence(self.run)

    def test_warmups_and_other_phases_excluded(self):
        pd.DataFrame([{"run_id": "fixture", "config_hash": self.configs["final"]["config_hash"],
                       "implementation_id": "final_python", "phase": phase, "seconds": seconds,
                       "trial": i, "sequence": i, "is_warmup": warmup}
                      for i, (phase, seconds, warmup) in enumerate([
                          ("prediction", 100, True), ("prediction", 1, False),
                          ("prediction", 3, False), ("fit", 999, False)])]).to_csv(self.run / "benchmark_raw.csv", index=False)
        timings = prediction_timings(self.run)
        self.assertEqual(timings.seconds.tolist(), [1, 3])
        self.assertEqual(float(timings.seconds.median()), 2)

    def test_saved_only_regeneration_is_deterministic(self):
        with patch("subprocess.run", side_effect=AssertionError("Report must not invoke external tools")):
            report = plot_results(self.run)
            first = {name: sha256(self.run / "figures" / name) for name in report["figures"]}
            regenerated = plot_results(self.run)
        self.assertEqual(len(report["figures"]), 8)
        self.assertTrue(report["omissions"])
        self.assertEqual(first, {name: sha256(self.run / "figures" / name) for name in regenerated["figures"]})
        self.assertIn("previously inspected", (self.run / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
