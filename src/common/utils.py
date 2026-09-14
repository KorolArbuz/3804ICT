import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}

    required = {"X", "y", "row_ids", "original_ids", "feature_names"}
    if not required.issubset(data):
        raise ValueError(f"NPZ is missing fields: {required - data.keys()}")
    if set(data) != required:
        raise ValueError(f"NPZ contains unexpected fields: {set(data) - required}")

    X = data["X"]
    y = data["y"]
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("Invalid processed array shapes or empty data")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Processed values must be finite and labels binary")
    if len(np.unique(data["row_ids"])) != len(y):
        raise ValueError("Duplicate row IDs in processed data")

    identifiers_do_not_match = len(data["original_ids"]) != len(y)
    features_do_not_match = len(data["feature_names"]) != X.shape[1]
    if identifiers_do_not_match or features_do_not_match:
        raise ValueError("Processed identifiers/features have inconsistent lengths")

    return data


def load_pair(train, test):
    training = load_npz(train)
    testing = load_npz(test)

    if not np.array_equal(training["feature_names"], testing["feature_names"]):
        raise ValueError("Train/test feature names differ")
    if set(training["row_ids"]) & set(testing["row_ids"]):
        raise ValueError("Training and test rows overlap")

    return training, testing
