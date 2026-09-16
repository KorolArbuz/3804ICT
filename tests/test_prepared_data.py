import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.common.experiment import load_configs, array_hash
from src.data.dataset import load_dataset, split_dataset
from src.preprocessing.features import (
    MONEY_COLUMNS,
    PAY_STATUS_COLUMNS,
    signed_log,
    safe_divide,
)
from src.preprocessing.transforms import StructuredPayTransformer, V2Preprocessor
from src.preprocessing.prepared import prepare


def sample_frame(rows=160):
    index = np.arange(rows, dtype=float)
    values = {column: 1000 + 17 * index for column in MONEY_COLUMNS}
    values["BILL_AMT1"] = -100 + 31 * index
    for offset, column in enumerate(PAY_STATUS_COLUMNS):
        values[column] = ((index.astype(int) + offset) % 6 - 2).astype(float)
    values.update(
        AGE=21 + index % 30,
        SEX=1 + index % 2,
        EDUCATION=1 + index % 3,
        MARRIAGE=1 + index % 2,
    )
    return pd.DataFrame(values)


class PreprocessingTests(unittest.TestCase):
    def test_signed_log_and_denominator(self):
        np.testing.assert_allclose(signed_log([-np.e + 1, 0, np.e - 1]), [-1, 0, 1])
        actual = safe_divide([4, 4, 4, 4], [2, 0, -1, np.nan])
        self.assertEqual(actual[0], 2)
        self.assertTrue(np.isnan(actual[1:]).all())

    def test_structured_pay_unseen_and_training_only(self):
        transform = StructuredPayTransformer(columns=("PAY_A", "PAY_B")).fit(
            [[-2, 0], [0, 1], [-1, 2]]
        )
        learned = [x.copy() for x in transform.categories_]
        result = transform.transform([[-99, -88]])
        positive = transform.transform([[3, 3]])
        np.testing.assert_array_equal(result[:, 2:], positive[:, 2:])
        for old, new in zip(learned, transform.categories_):
            np.testing.assert_array_equal(old, new)
        self.assertTrue(np.isfinite(result).all())

    def test_feature_order_fit_statistics_and_group_weight(self):
        config = load_configs()["final"]["model"]
        frame = sample_frame()
        first = V2Preprocessor(SimpleNamespace(**config)).fit(frame.iloc[:100])
        means = (
            first.transformer_.named_transformers_["money"]
            .named_steps["scaler"]
            .mean_.copy()
        )
        first.transform(frame.iloc[100:] * 100)
        np.testing.assert_array_equal(
            means,
            first.transformer_.named_transformers_["money"].named_steps["scaler"].mean_,
        )
        self.assertEqual(
            len(set(first.get_feature_names_out())), len(first.get_feature_names_out())
        )
        one = first.transform(frame)
        two = (
            V2Preprocessor(SimpleNamespace(**{**config, "pay_group_weight": 2.0}))
            .fit(frame.iloc[:100])
            .transform(frame)
        )
        np.testing.assert_allclose(
            two[:, first.pay_group_indices_],
            one[:, first.pay_group_indices_] * np.sqrt(2),
        )
        other = np.setdiff1d(np.arange(one.shape[1]), first.pay_group_indices_)
        np.testing.assert_array_equal(two[:, other], one[:, other])

    def test_two_groups_identical_split_and_lossless_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = sample_frame()
            frame.insert(0, "ID", np.arange(len(frame)) + 1)
            frame["target"] = np.arange(len(frame)) % 2
            data = root / "data.csv"
            frame.to_csv(data, index=False)
            groups, manifest = prepare(data, root / "prepared/data", load_configs())
            left, right = groups["baseline"], groups["final"]
            np.testing.assert_array_equal(left["row_ids"], right["row_ids"])
            np.testing.assert_array_equal(left["y_test"], right["y_test"])
            self.assertFalse(set(left["train_ids"]) & set(left["row_ids"]))
            self.assertEqual(
                array_hash(left["row_ids"]), manifest["split"]["test_row_ids_hash"]
            )
            for group in groups.values():
                lines = (
                    (group["directory"] / "test.arff")
                    .read_text()
                    .split("@DATA\n")[1]
                    .splitlines()
                )
                self.assertTrue(all(line.endswith(",?") for line in lines))
                self.assertFalse((group["directory"] / "test_labels.csv").exists())


if __name__ == "__main__":
    unittest.main()
