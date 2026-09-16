"""Dataset-free subprocess failure contracts; no compiler or JVM is launched."""

import io
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.cpp_knn import bridge
from src.weka_bridge import runner as weka


class ArtifactFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_cpp_executable_does_not_start_process(self):
        with patch.object(bridge.subprocess, "run") as launch:
            with self.assertRaisesRegex(RuntimeError, "missing"):
                bridge.validate_executable(self.directory / "missing.exe")
            launch.assert_not_called()

    def test_stale_cpp_identity_is_rejected(self):
        executable = self.directory / "cpp with spaces.exe"
        executable.write_bytes(b"synthetic fixture; never executable")
        reply = subprocess.CompletedProcess([], 0, json.dumps({"source_id": "old"}), "")
        with patch.object(bridge.subprocess, "run", return_value=reply) as launch:
            with self.assertRaisesRegex(RuntimeError, "stale"):
                bridge.validate_executable(executable)
            self.assertEqual(launch.call_args.args[0], [str(executable), "--identity"])
            self.assertTrue(launch.call_args.kwargs["check"])
            self.assertGreater(launch.call_args.kwargs["timeout"], 0)

    def test_cpp_identity_nonzero_and_malformed_are_rejected(self):
        executable = self.directory / "fake.exe"
        executable.write_bytes(b"not executed")
        with patch.object(
            bridge.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(9, [str(executable)]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                bridge.validate_executable(executable)
        reply = subprocess.CompletedProcess([], 0, "not-json", "")
        with patch.object(bridge.subprocess, "run", return_value=reply):
            with self.assertRaises(json.JSONDecodeError):
                bridge.validate_executable(executable)

    def test_cpp_wrong_build_configuration_is_rejected(self):
        executable = self.directory / "debug.exe"
        executable.write_bytes(b"fixture")
        reply = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "source_id": "current",
                    "build_mode": "portable",
                    "build_configuration": "Debug",
                }
            ),
            "",
        )
        with patch.object(bridge, "source_id", return_value="current"), patch.object(
            bridge.subprocess, "run", return_value=reply
        ):
            with self.assertRaisesRegex(RuntimeError, "Release"):
                bridge.validate_executable(executable)

    def test_missing_weka_artifact_is_rejected(self):
        with patch.object(weka, "JAR", self.directory / "missing.jar"), patch.object(
            weka, "BUILD_MANIFEST", self.directory / "missing.json"
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                weka.verify_artifact()

    def test_stale_or_modified_weka_jar_is_rejected(self):
        jar = self.directory / "model.jar"
        manifest = self.directory / "build_manifest.json"
        jar.write_bytes(b"jar fixture")
        manifest.write_text(
            json.dumps(
                {"source_hashes": {"source": "old"}, "artifact_sha256": weka._sha(jar)}
            ),
            encoding="utf-8",
        )
        with patch.object(weka, "JAR", jar), patch.object(
            weka, "BUILD_MANIFEST", manifest
        ), patch.object(weka, "source_hashes", return_value={"source": "new"}):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                weka.verify_artifact()
        manifest.write_text(
            json.dumps(
                {
                    "source_hashes": {"source": "current"},
                    "artifact_sha256": "wrong bytes",
                }
            ),
            encoding="utf-8",
        )
        with patch.object(weka, "JAR", jar), patch.object(
            weka, "BUILD_MANIFEST", manifest
        ), patch.object(weka, "source_hashes", return_value={"source": "current"}):
            with self.assertRaisesRegex(RuntimeError, "modified"):
                weka.verify_artifact()


class WorkerFailureTests(unittest.TestCase):
    @staticmethod
    def cpp_worker():
        worker = bridge.PersistentCpp.__new__(bridge.PersistentCpp)
        worker.process = MagicMock()
        worker.timeout = 0.01
        worker._lines = queue.Queue()
        worker._stderr = io.StringIO("synthetic native error")
        worker.close = MagicMock()
        return worker

    @staticmethod
    def weka_worker():
        worker = weka.WekaWorker.__new__(weka.WekaWorker)
        worker.process = MagicMock()
        worker.timeout = 0.01
        worker._queue = queue.Queue()
        worker.ready = {"n_query": 2}
        worker._checksum = None
        worker.close = MagicMock()
        return worker

    def test_cpp_timeout_closes_worker(self):
        worker = self.cpp_worker()
        with patch.object(worker._lines, "get", side_effect=queue.Empty):
            with self.assertRaises(TimeoutError):
                worker.predict()
        worker.close.assert_called_once()

    def test_cpp_early_exit_closes_worker(self):
        worker = self.cpp_worker()
        worker.process.poll.return_value = 9
        worker._lines.put(None)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.predict()
        worker.close.assert_called_once()

    def test_cpp_malformed_response_closes_worker(self):
        for line in (
            "garbage",
            '{"status":"WRONG"}',
            '{"status":"PREDICTION","seconds":-1}',
        ):
            with self.subTest(line=line):
                worker = self.cpp_worker()
                worker._lines.put(line)
                with self.assertRaises((ValueError, RuntimeError)):
                    worker.predict()
                worker.close.assert_called_once()

    def test_weka_timeout_and_early_exit_close_worker(self):
        worker = self.weka_worker()
        with patch.object(worker._queue, "get", side_effect=queue.Empty):
            with self.assertRaises(TimeoutError):
                worker.predict()
        worker.close.assert_called_once()
        worker = self.weka_worker()
        worker._queue.put(None)
        with self.assertRaisesRegex(RuntimeError, "exited"):
            worker.predict()
        worker.close.assert_called_once()

    def test_weka_malformed_query_count_closes_worker(self):
        worker = self.weka_worker()
        worker._queue.put(
            json.dumps(
                {"status": "PREDICT", "seconds": 0.1, "n_query": 3, "checksum": "x"}
            )
        )
        with self.assertRaisesRegex(RuntimeError, "query count"):
            worker.predict()
        worker.close.assert_called_once()

    def test_weka_changed_checksum_closes_worker(self):
        worker = self.weka_worker()
        worker._checksum = "first"
        worker._queue.put(
            json.dumps(
                {
                    "status": "PREDICT",
                    "seconds": 0.1,
                    "n_query": 2,
                    "checksum": "second",
                }
            )
        )
        with self.assertRaisesRegex(RuntimeError, "changed"):
            worker.predict()
        worker.close.assert_called_once()

    def test_weka_timeout_kills_process_tree(self):
        process = MagicMock()
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["java"], 1),
            ("", ""),
        ]
        process.__enter__.return_value = process
        with patch.object(weka.subprocess, "Popen", return_value=process), patch.object(
            weka, "_kill_tree"
        ) as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                weka._run(["java", "synthetic"], timeout=1)
        kill.assert_called_once_with(process)
        self.assertEqual(process.communicate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
