"""Strict prediction contract and one evaluator for all five implementations."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common.experiment import SCORE_TOLERANCE, array_hash, model_group
from src.common.utils import write_json

COLUMNS = [
    "run_id",
    "implementation_id",
    "model_group",
    "config_hash",
    "test_position",
    "row_id",
    "original_id",
    "y_true",
    "score_class_1",
    "y_pred",
    "decision_rule",
    "threshold",
]


def prediction_frame(run_id, implementation, config, data, scores, labels):
    if np.asarray(scores).shape != (len(data["y_test"]),) or np.asarray(
        labels
    ).shape != (len(data["y_test"]),):
        raise ValueError("Scores/labels have incorrect shape")
    return pd.DataFrame(
        {
            "run_id": run_id,
            "implementation_id": implementation,
            "model_group": model_group(implementation),
            "config_hash": config["config_hash"],
            "test_position": np.arange(len(scores)),
            "row_id": data["row_ids"],
            "original_id": data["original_ids"],
            "y_true": data["y_test"],
            "score_class_1": scores,
            "y_pred": labels,
            "decision_rule": config["model"]["decision_rule"],
            "threshold": config["model"]["threshold"],
        },
        columns=COLUMNS,
    )


def validate_predictions(frame, run_id, implementation, config, data_manifest):
    if list(frame.columns) != COLUMNS or len(frame) == 0:
        raise ValueError("Prediction schema/empty data")
    constants = {
        "run_id": run_id,
        "implementation_id": implementation,
        "model_group": model_group(implementation),
        "config_hash": config["config_hash"],
        "decision_rule": config["model"]["decision_rule"],
    }
    for column, value in constants.items():
        if not (frame[column] == value).all():
            raise ValueError(f"Stale or incorrect prediction identity: {column}")
    if (
        not np.array_equal(frame.test_position, np.arange(len(frame)))
        or frame.row_id.duplicated().any()
    ):
        raise ValueError("Prediction order or duplicate IDs")
    if not np.array_equal(frame.row_id, frame.row_id.astype(np.int64)):
        raise ValueError("Nonintegral row IDs")
    split = data_manifest["split"]
    for values, key in [
        (frame.row_id.to_numpy(dtype=np.int64), "test_row_ids_hash"),
        (frame.y_true.to_numpy(dtype=np.int64), "y_test_hash"),
        (frame.original_id.astype(str).to_numpy(), "original_test_ids_hash"),
    ]:
        if array_hash(values) != split[key]:
            raise ValueError(f"Prediction identity/ground truth mismatch: {key}")
    scores = frame.score_class_1.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Nonfinite/out-of-range scores")
    if (
        not np.isin(frame.y_true, [0, 1]).all()
        or not np.isin(frame.y_pred, [0, 1]).all()
    ):
        raise ValueError("Nonbinary labels")
    threshold = config["model"]["threshold"]
    if threshold is None:
        if not frame.threshold.isna().all():
            raise ValueError("Baseline threshold must be null")
        expected = scores > 0.5
    else:
        if not (frame.threshold == threshold).all():
            raise ValueError("Incorrect threshold")
        expected = scores >= threshold
    if not np.array_equal(frame.y_pred, expected.astype(int)):
        raise ValueError("Labels inconsistent with decision rule/scores")
    return frame


def read_predictions(path):
    return pd.read_csv(
        path,
        dtype={"original_id": str, "config_hash": str},
        float_precision="round_trip",
        keep_default_na=False,
        na_values={"threshold": [""]},
    )


def metrics(frame):
    y = frame.y_true.to_numpy()
    prediction = frame.y_pred.to_numpy()
    score = frame.score_class_1.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "implementation_id": frame.implementation_id.iloc[0],
        "model_group": frame.model_group.iloc[0],
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "specificity": float(tn / (tn + fp)),
        "f1": f1_score(y, prediction, zero_division=0),
        "f0_5": fbeta_score(y, prediction, beta=0.5, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "predicted_positive_rate": float(prediction.mean()),
        "false_positive_rate": float(fp / (tn + fp)),
    }


def evaluate(run_dir, run_id, implementations, configs, data_manifest):
    run_dir = Path(run_dir)
    frames = {
        name: validate_predictions(
            read_predictions(run_dir / "predictions" / f"{name}.csv"),
            run_id,
            name,
            configs[model_group(name)],
            data_manifest,
        )
        for name in implementations
    }
    table = pd.DataFrame([metrics(frame) for frame in frames.values()])
    table.to_csv(run_dir / "metrics.csv", index=False, float_format="%.17g")
    agreements = []
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for a, b in combinations([name for name in frames if name.startswith("final_")], 2):
        left, right = frames[a], frames[b]
        delta = np.abs(left.score_class_1.to_numpy() - right.score_class_1.to_numpy())
        changed = left.y_pred.to_numpy() != right.y_pred.to_numpy()
        selected = (delta > SCORE_TOLERANCE) | changed
        agreements.append(
            {
                "implementation_a": a,
                "implementation_b": b,
                "rows": len(left),
                "label_disagreements": int(changed.sum()),
                "label_agreement": float(1 - changed.mean()),
                "max_abs_score_difference": float(delta.max()),
                "mean_abs_score_difference": float(delta.mean()),
                "score_tolerance": SCORE_TOLERANCE,
                "scores_over_tolerance": int((delta > SCORE_TOLERANCE).sum()),
            }
        )
        evidence = left.loc[
            selected, ["test_position", "row_id", "original_id", "y_true"]
        ].copy()
        for name, frame in [(a, left), (b, right)]:
            evidence[name + "_score"] = frame.loc[selected, "score_class_1"]
            evidence[name + "_label"] = frame.loc[selected, "y_pred"]
            evidence[name + "_threshold_margin"] = (
                frame.loc[selected, "score_class_1"]
                - configs["final"]["model"]["threshold"]
            )
        evidence["abs_score_difference"] = delta[selected]
        evidence.to_csv(
            diagnostics / f"{a}__{b}.csv", index=False, float_format="%.17g"
        )
    agreement = pd.DataFrame(
        agreements,
        columns=[
            "implementation_a",
            "implementation_b",
            "rows",
            "label_disagreements",
            "label_agreement",
            "max_abs_score_difference",
            "mean_abs_score_difference",
            "score_tolerance",
            "scores_over_tolerance",
        ],
    )
    agreement.to_csv(run_dir / "agreement.csv", index=False, float_format="%.17g")
    write_json(
        diagnostics / "interpretation.json",
        {
            "score_tolerance": SCORE_TOLERANCE,
            "custom_cpp": "same intended semantics; any above-tolerance difference requires row-level investigation",
            "sklearn": "native neighbour selection/distance reduction, not assumed identical to manual kernel",
            "weka": "native IBk distribution includes its distance weighting, tie and prior semantics; labels use the same frozen threshold",
            "validation_scope": "all fixed test rows; no new model or threshold selection",
        },
    )
    return table, agreement, frames
