from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.utils import load_npz, read_json


def _safe_divide(numerator, denominator):
    if denominator:
        return numerator / denominator
    return float("nan")


def confusion_counts(y_true, y_pred):
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    return {
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
    }


def classification_metrics(y_true, y_pred, probabilities=None):
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Labels must be nonempty, aligned one-dimensional arrays")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1")

    counts = confusion_counts(actual, predicted)
    tn, fp, fn, tp = (counts[name] for name in ("TN", "FP", "FN", "TP"))
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)

    result = {
        "test_samples": len(actual),
        **counts,
        "accuracy": (tn + tp) / len(actual),
        "precision": _safe_divide(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
    }

    if probabilities is not None:
        probability_values = np.asarray(probabilities, dtype=float)
        if (
            probability_values.shape != actual.shape
            or not np.isfinite(probability_values).all()
            or ((probability_values < 0) | (probability_values > 1)).any()
        ):
            raise ValueError("Probabilities must be finite, aligned and in [0,1]")

        if len(np.unique(actual)) == 2:
            result["roc_auc"] = float(roc_auc_score(actual, probability_values))
        else:
            result["roc_auc"] = float("nan")

        if (actual == 1).any():
            result["average_precision"] = float(
                average_precision_score(actual, probability_values)
            )
        else:
            result["average_precision"] = float("nan")

    return result


def validate_predictions(path, test):
    frame = pd.read_csv(path, dtype={"original_id": str}, keep_default_na=False)
    required = {
        "test_index",
        "row_id",
        "original_id",
        "y_true",
        "y_pred",
        "probability_class_1",
        "selected_k",
    }
    if not required.issubset(frame):
        raise ValueError(f"Prediction file lacks {required - set(frame)}")
    if frame.row_id.duplicated().any() or frame.test_index.duplicated().any():
        raise ValueError("Duplicate prediction row IDs or test indexes")
    if len(frame) != len(test["y"]):
        raise ValueError("Prediction row count differs from test set")

    expected_columns = (
        ("test_index", np.arange(len(test["y"]))),
        ("row_id", test["row_ids"]),
        ("original_id", test["original_ids"]),
        ("y_true", test["y"]),
    )
    for column, expected in expected_columns:
        if not np.array_equal(frame[column].to_numpy(), expected):
            raise ValueError(f"Prediction {column} alignment mismatch")

    if frame.selected_k.nunique() != 1 or frame.selected_k.iloc[0] < 1:
        raise ValueError("Prediction k is invalid or inconsistent")

    classification_metrics(frame.y_true, frame.y_pred, frame.probability_class_1)
    return frame


def evaluate_file(path, test_path, implementation=None):
    test = load_npz(test_path)
    frame = validate_predictions(path, test)
    runtime_path = Path(path).with_suffix(".runtime.json")
    timing = read_json(runtime_path)
    if timing["selected_k"] != int(frame.selected_k.iloc[0]):
        raise ValueError("Runtime and predictions use different k values")
    for key in ("fit_seconds", "prediction_seconds", "milliseconds_per_sample"):
        if not np.isfinite(timing[key]) or timing[key] < 0:
            raise ValueError(f"Invalid measured timing: {key}")

    result = {
        "implementation": implementation or timing["implementation"],
        "selected_k": timing["selected_k"],
        **classification_metrics(
            frame.y_true,
            frame.y_pred,
            frame.probability_class_1,
        ),
    }
    for key in ("fit_seconds", "prediction_seconds", "milliseconds_per_sample"):
        result[key] = timing[key]

    return result
