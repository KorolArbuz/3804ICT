"""Aggregation and deterministic training-CV model selection."""

from pathlib import Path

import pandas as pd

from src.common.config import TIE_TOLERANCE


QUALITY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
)
GROUP_COLUMNS = ("k", "metric", "weights")


def aggregate_cv(results):
    """Return one mean/std/count row for each candidate configuration."""
    required = {"fold", *GROUP_COLUMNS, *QUALITY_METRICS, "TN", "FP", "FN", "TP"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"CV results lack columns: {sorted(missing)}")

    grouped = results.groupby(list(GROUP_COLUMNS), sort=False)[list(QUALITY_METRICS)]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    count_columns = [f"{metric}_count" for metric in QUALITY_METRICS]
    if not summary[count_columns].eq(5).all().all():
        raise ValueError("Every configuration must contain exactly five folds")
    return summary


def select_configuration(summary, tolerance=TIE_TOLERANCE):
    """Apply the predeclared F1-first selection and deterministic tie-breaks."""
    if len(summary) != 64:
        raise ValueError(f"Expected 64 aggregated configurations; found {len(summary)}")

    candidates = summary[summary.f1_mean >= summary.f1_mean.max() - tolerance]
    candidates = candidates[
        candidates.balanced_accuracy_mean
        >= candidates.balanced_accuracy_mean.max() - tolerance
    ]
    candidates = candidates[
        candidates.recall_mean >= candidates.recall_mean.max() - tolerance
    ]
    candidates = candidates[candidates.k == candidates.k.min()].copy()
    candidates["metric_order"] = candidates.metric.map(
        {"euclidean": 0, "manhattan": 1}
    )
    candidates["weights_order"] = candidates.weights.map(
        {"uniform": 0, "distance": 1}
    )
    if candidates[["metric_order", "weights_order"]].isna().any().any():
        raise ValueError("Unexpected metric or voting rule in selection table")
    winner = candidates.sort_values(["metric_order", "weights_order"]).iloc[0]
    return winner.drop(labels=["metric_order", "weights_order"])


def configuration_row(summary, k, metric, weights):
    matches = summary[
        (summary.k == int(k))
        & (summary.metric == metric)
        & (summary.weights == weights)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row for k={k}, {metric}, {weights}")
    return matches.iloc[0]


def row_to_dict(row):
    result = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        result[str(key)] = value
    return result


def require_frozen_selection(path):
    """Load and validate the parameters frozen by training-only CV."""
    import json

    selection_path = Path(path)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Run training-only search first; selection is missing: {selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    required = {
        "selected_k",
        "metric",
        "weights",
        "selection_metric",
        "selection_rule",
        "test_evaluated",
        "cv_results_sha256",
    }
    missing = required - set(selection)
    if missing:
        raise ValueError(f"Selection artifact lacks fields: {sorted(missing)}")
    if selection["metric"] not in {"euclidean", "manhattan"}:
        raise ValueError("Selection contains an unsupported distance metric")
    if selection["weights"] not in {"uniform", "distance"}:
        raise ValueError("Selection contains an unsupported voting rule")
    if int(selection["selected_k"]) < 1:
        raise ValueError("Selection contains an invalid k")
    return selection
