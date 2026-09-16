"""Exact, deterministic threshold selection for training-only OOF scores."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import classification_metrics


THRESHOLD_RULE = (
    "max F1; max balanced accuracy; max recall; closest to 0.5; larger threshold"
)


def threshold_grid(y_true, scores):
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    thresholds = np.unique(np.r_[scores, 0.0, 0.5, 1.0, np.nextafter(1.0, np.inf)])
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    positive_prefix = np.r_[0, np.cumsum(labels[order], dtype=np.int64)]
    below = np.searchsorted(sorted_scores, thresholds, side="left")
    fn = positive_prefix[below]
    tn = below - fn
    tp = int(labels.sum()) - fn
    fp = len(labels) - int(labels.sum()) - tn

    def divide(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float), where=b != 0)

    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return pd.DataFrame({
        "threshold": thresholds, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": (tp + tn) / len(labels), "precision": divide(tp, tp + fp),
        "recall": recall, "specificity": specificity,
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "predicted_positive_rate": (tp + fp) / len(labels),
    })


def select_f1_threshold(y_true, scores):
    grid = threshold_grid(y_true, scores)
    ranked = grid.assign(distance_to_default=(grid.threshold - 0.5).abs()).sort_values(
        ["f1", "balanced_accuracy", "recall", "distance_to_default", "threshold"],
        ascending=[False, False, False, True, False], kind="stable",
    )
    best = ranked.iloc[0]
    return float(best.threshold), {
        key: (int(best[key]) if key in {"TP", "TN", "FP", "FN"} else float(best[key]))
        for key in ("accuracy", "precision", "recall", "specificity", "f1",
                    "balanced_accuracy", "TP", "TN", "FP", "FN",
                    "predicted_positive_rate")
    }


def score_metrics(y_true, scores, threshold):
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    result = classification_metrics(labels, predictions, scores)
    result.pop("test_samples")
    result["threshold"] = float(threshold)
    result["predicted_positive_rate"] = float(predictions.mean())
    return result


def ranking_metrics(y_true, scores):
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
