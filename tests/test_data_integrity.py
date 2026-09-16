import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.common.experiment import IMPLEMENTATIONS, load_configs
from src.validation import runner as validation


class PreparedIntegrityTests(unittest.TestCase):
    def test_permuted_rows_or_columns_fail_saved_input_identity(self):
        original = np.array([[1.125, 2.5], [3.25, 4.875]])
        for altered in (original[::-1], original[:, ::-1]):
            with self.subTest(
                altered=altered.tolist()
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run = root / "run"
                directory = run / "data/baseline"
                directory.mkdir(parents=True)
                (root / "docs").mkdir()
                (root / "docs/migration_characterization.json").write_text("{}")
                path = directory / "train_features.csv"
                np.savetxt(path, original, delimiter=",", fmt="%.17g")
                expected = hashlib.sha256(path.read_bytes()).hexdigest()
                (run / "run_manifest.json").write_text(
                    json.dumps({"completed_implementations": list(IMPLEMENTATIONS)})
                )
                (run / "resolved_configs.json").write_text(json.dumps(load_configs()))
                (run / "data_manifest.json").write_text(
                    json.dumps(
                        {
                            "groups": {
                                "baseline": {"files": {"train_features.csv": expected}}
                            }
                        }
                    )
                )
                np.savetxt(path, altered, delimiter=",", fmt="%.17g")
                with patch.object(validation, "ROOT", root), self.assertRaisesRegex(
                    ValueError, "Prepared file hash changed"
                ):
                    validation.validate_saved_run(run)


if __name__ == "__main__":
    unittest.main()
