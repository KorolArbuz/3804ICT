import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from src.benchmarking.suite import summarize
from src.benchmarking.evidence import paired_ratios
from src.benchmarking.suite import benchmark
from src.common.experiment import IMPLEMENTATIONS


class BenchmarkProtocolTests(unittest.TestCase):
    def test_summary_keeps_outliers_excludes_warmup_and_separates_baseline(self):
        rows = []
        for name, values in [
            ("baseline_python_v5_1", [1, 2, 100]),
            ("final_python", [2, 4, 200]),
            ("final_cpp", [1, 2, 100]),
        ]:
            for trial, seconds in enumerate(values):
                rows.append(
                    dict(
                        implementation_id=name,
                        phase="prediction",
                        seconds=seconds,
                        is_warmup=False,
                        run_id="session",
                        trial=trial,
                        config_hash="baseline"
                        if name.startswith("baseline")
                        else "final",
                        n_train=100,
                        n_query=20,
                        n_features=2,
                        k=19 if name.startswith("baseline") else 101,
                    )
                )
            rows.append({**rows[-1], "seconds": 99999, "is_warmup": True})
        result = summarize(pd.DataFrame(rows)).set_index("implementation_id")
        self.assertEqual(result.loc["final_cpp", "max_seconds"], 100)
        self.assertEqual(result.loc["final_cpp", "n"], 3)
        self.assertEqual(
            result.loc["final_cpp", "std_seconds"], np.std([1, 2, 100], ddof=1)
        )
        self.assertEqual(result.loc["final_cpp", "final_python_over_implementation"], 2)
        self.assertTrue(
            np.isnan(
                result.loc["baseline_python_v5_1", "final_python_over_implementation"]
            )
        )

    def test_ratios_are_paired_before_aggregation(self):
        rows = []
        for name, values in [
            ("final_python", [1, 100, 100]),
            ("final_cpp", [1, 2, 100]),
        ]:
            for trial, seconds in enumerate(values):
                rows.append(
                    dict(
                        run_id="one",
                        trial=trial,
                        config_hash="same",
                        n_train=100,
                        n_query=20,
                        n_features=2,
                        k=101,
                        implementation_id=name,
                        phase="prediction",
                        seconds=seconds,
                        is_warmup=False,
                    )
                )
        result = summarize(pd.DataFrame(rows)).set_index("implementation_id")
        self.assertEqual(result.loc["final_cpp", "final_python_over_implementation"], 1)
        self.assertEqual(result.loc["final_cpp", "paired_trial_count"], 3)
        changed = pd.DataFrame(rows)
        changed.loc[
            changed.implementation_id == "final_cpp", "run_id"
        ] = "different-session"
        self.assertFalse(
            (paired_ratios(changed).implementation_id == "final_cpp").any()
        )

    def test_duplicate_trial_keys_rejected(self):
        row = dict(
            run_id="one",
            trial=0,
            config_hash="same",
            n_train=100,
            n_query=20,
            n_features=2,
            k=101,
            implementation_id="final_python",
            phase="prediction",
            seconds=1.0,
            is_warmup=False,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_ratios(pd.DataFrame([row, row]))

    def test_complete_coordinator_contract_with_mocked_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "predictions").mkdir()
            payload = "test_position,score_class_1,y_pred\n0,0.1,0\n1,0.9,1\n"
            for name in IMPLEMENTATIONS:
                (directory / "predictions" / f"{name}.csv").write_text(
                    payload, encoding="utf-8"
                )
            (directory / "run_manifest.json").write_text(
                json.dumps({"sources": {}}), encoding="utf-8"
            )
            (directory / "environment.json").write_text(
                json.dumps(
                    {
                        "python": "test",
                        "packages": {},
                        "thread_environment": {},
                        "threadpools": [],
                    }
                ),
                encoding="utf-8",
            )
            data = {
                "X_train": np.zeros((109, 2)),
                "X_test": np.zeros((2, 2)),
                "directory": directory,
            }
            configs = {
                group: {
                    "config_hash": group,
                    "model": {
                        "k": 19 if group == "baseline" else 101,
                        "threshold": None
                        if group == "baseline"
                        else 0.3315411365543412,
                    },
                }
                for group in ("baseline", "final")
            }
            models = {}
            for name in IMPLEMENTATIONS[:3]:
                model = MagicMock()
                model.fit_seconds = 0.01
                model.measured_predict.return_value = (
                    {"seconds": 0.1, "checksum": "stable"},
                    np.array([0.1, 0.9]),
                    np.array([0, 1]),
                )
                models[name] = model
            cpp, java = MagicMock(), MagicMock()
            for worker in (cpp, java):
                worker.__enter__.return_value = worker
                worker.ready = {
                    "fit_seconds": 0.02,
                    "artifact_sha256": "jar-hash",
                    "compiler": "fixture",
                    "flags": "fixture",
                }
                worker.identity = {"executable_sha256": "exe-hash"}
                worker.process.pid = 1234
                worker.predict.return_value = {"seconds": 0.2, "checksum": "stable"}

            def launch(command, **kwargs):
                key = "--predictions" if "--predictions" in command else "--output"
                output = Path(command[command.index(key) + 1])
                output.write_text(payload, encoding="utf-8")
                process = MagicMock()
                process.pid = 9876
                process.returncode = 0
                process.communicate.return_value = ("", "")
                return process

            with patch("src.cpp_knn.bridge.PersistentCpp", return_value=cpp), patch(
                "src.cpp_knn.bridge.command_arguments", return_value=["fake-cpp"]
            ), patch("src.weka_bridge.runner.WekaWorker", return_value=java), patch(
                "src.weka_bridge.runner.prediction_command",
                side_effect=lambda a, b, output, **kw: [
                    "fake-java",
                    "--output",
                    str(output),
                ],
            ), patch(
                "src.benchmarking.suite.subprocess.Popen", side_effect=launch
            ), patch(
                "src.benchmarking.suite.sha256", return_value="artifact-hash"
            ), patch(
                "src.benchmarking.evidence.process_affinity",
                return_value={"cpus": [0], "status": "observed"},
            ):
                benchmark(
                    directory,
                    list(IMPLEMENTATIONS),
                    {"baseline": data, "final": data},
                    configs,
                    models,
                    Path("fake.exe"),
                    "portable",
                    runs=2,
                    warmups=1,
                )
            raw = pd.read_csv(directory / "benchmark_raw.csv")
            self.assertEqual(len(raw), 40)
            self.assertTrue(
                {
                    "artifact_sha256",
                    "implementation_source_sha256",
                    "knn_compute_threads",
                    "affinity",
                    "process_mode",
                }.issubset(raw.columns)
            )
            pairs = pd.read_csv(directory / "benchmark_paired_ratios.csv")
            self.assertEqual(len(pairs), 28)
            self.assertFalse(pairs.implementation_id.str.startswith("baseline").any())
            self.assertEqual(len(pd.read_csv(directory / "benchmark_setup.csv")), 5)
            self.assertTrue(
                raw.loc[raw.phase == "prepared_pipeline", "checksum"]
                .str.startswith("sha256:")
                .all()
            )


if __name__ == "__main__":
    unittest.main()
