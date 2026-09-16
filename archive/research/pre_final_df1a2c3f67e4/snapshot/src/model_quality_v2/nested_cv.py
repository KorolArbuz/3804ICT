"""Leakage-safe 5x3 nested-CV building blocks for Model Quality V2."""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .thresholding import ranking_metrics, score_metrics, select_f1_threshold
from .transforms import V2Preprocessor

OUTER_FOLDS = 5
INNER_FOLDS = 3
RANDOM_STATE = 42
METRICS = ("average_precision", "roc_auc", "f1", "recall", "precision",
           "balanced_accuracy")


def deterministic_splits(y, folds, seed=RANDOM_STATE):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    placeholder = np.zeros(len(y))
    return [(fit.copy(), validation.copy()) for fit, validation in splitter.split(placeholder, y)]


def _distance_scores(distances, labels, k):
    distances = distances[:, :k]
    labels = labels[:, :k]
    zero = distances == 0
    has_zero = zero.any(axis=1)
    result = np.empty(len(distances), dtype=np.float64)
    if has_zero.any():
        result[has_zero] = np.divide(
            np.sum(labels[has_zero] * zero[has_zero], axis=1),
            np.sum(zero[has_zero], axis=1),
        )
    if (~has_zero).any():
        inverse = 1.0 / distances[~has_zero]
        result[~has_zero] = np.sum(inverse * labels[~has_zero], axis=1) / inverse.sum(axis=1)
    return result


def _score_configuration_group(X_fit, y_fit, X_validation, configurations):
    """Run one neighbour query for configurations sharing a representation."""
    maximum_k = max(config.k for config in configurations)
    model = KNeighborsClassifier(
        n_neighbors=maximum_k, metric="euclidean", weights="distance",
        algorithm="brute", n_jobs=1,
    ).fit(X_fit, y_fit)
    distances, indices = model.kneighbors(X_validation, n_neighbors=maximum_k)
    neighbour_labels = y_fit[indices]
    return {
        config.configuration_id: _distance_scores(distances, neighbour_labels, config.k)
        for config in configurations
    }


def inner_oof_selection(X, y, configurations, outer_fold, stage, seed=RANDOM_STATE):
    """Select a configuration and threshold using only the supplied training rows."""
    configurations = list(configurations)
    scores = {config.configuration_id: np.full(len(y), np.nan) for config in configurations}
    detailed = []
    groups = defaultdict(list)
    for config in configurations:
        groups[config.representation_key].append(config)
    with threadpool_limits(limits=1):
        for inner_fold, (fit, validation) in enumerate(deterministic_splits(y, INNER_FOLDS, seed), 1):
            for group in groups.values():
                preprocessor = V2Preprocessor(group[0])
                X_fit = preprocessor.fit_transform(X.iloc[fit], y[fit])
                X_validation = preprocessor.transform(X.iloc[validation])
                fold_scores = _score_configuration_group(X_fit, y[fit], X_validation, group)
                for config in group:
                    values = fold_scores[config.configuration_id]
                    scores[config.configuration_id][validation] = values
                    rank = ranking_metrics(y[validation], values)
                    detailed.append({
                        "stage": stage, "outer_fold": outer_fold, "inner_fold": inner_fold,
                        "configuration_id": config.configuration_id, "row_type": "fold",
                        "validation_rows": len(validation), **rank,
                    })
    summaries = []
    for config in configurations:
        values = scores[config.configuration_id]
        if not np.isfinite(values).all():
            raise ValueError("Every inner-training row must receive exactly one OOF score")
        folds = [row for row in detailed if row["configuration_id"] == config.configuration_id]
        threshold, threshold_values = select_f1_threshold(y, values)
        summary = {
            "stage": stage, "outer_fold": outer_fold, "inner_fold": 0,
            "configuration_id": config.configuration_id, "row_type": "summary",
            "validation_rows": len(y),
            "average_precision": float(np.mean([row["average_precision"] for row in folds])),
            "roc_auc": float(np.mean([row["roc_auc"] for row in folds])),
            "oof_average_precision": ranking_metrics(y, values)["average_precision"],
            "oof_roc_auc": ranking_metrics(y, values)["roc_auc"],
            "selected_threshold": threshold, **threshold_values,
            "k": config.k,
        }
        summaries.append(summary)
    ranked = sorted(
        summaries,
        key=lambda row: (-row["average_precision"], -row["roc_auc"], -row["f1"],
                         -row["balanced_accuracy"], row["k"], row["configuration_id"]),
    )
    selected_row = ranked[0]
    selected_id = selected_row["configuration_id"]
    for row in summaries:
        row["selected"] = row["configuration_id"] == selected_id
    selected = next(config for config in configurations if config.configuration_id == selected_id)
    return selected, selected_row["selected_threshold"], pd.DataFrame(detailed + summaries), scores


def evaluate_outer_configuration(X_fit, y_fit, X_validation, y_validation,
                                 configuration, threshold, stage, outer_fold, role):
    preprocessor = V2Preprocessor(configuration)
    transformed_fit = preprocessor.fit_transform(X_fit, y_fit)
    transformed_validation = preprocessor.transform(X_validation)
    model = KNeighborsClassifier(
        n_neighbors=configuration.k, metric="euclidean", weights="distance",
        algorithm="brute", n_jobs=1,
    ).fit(transformed_fit, y_fit)
    scores = model.predict_proba(transformed_validation)[:, 1]
    return {
        "outer_fold": outer_fold, "stage": stage, "role": role,
        "configuration_id": configuration.configuration_id,
        **score_metrics(y_validation, scores, threshold),
    }


def stage_acceptance(baseline_rows, candidate_rows):
    baseline = pd.DataFrame(baseline_rows).sort_values("outer_fold")
    candidate = pd.DataFrame(candidate_rows).sort_values("outer_fold")
    if not np.array_equal(baseline.outer_fold, candidate.outer_fold) or len(candidate) != OUTER_FOLDS:
        raise ValueError("Acceptance requires paired results for all five outer folds")
    report = {}
    for metric in METRICS:
        deltas = candidate[metric].to_numpy() - baseline[metric].to_numpy()
        report[metric] = {
            "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
            "wins": int(np.sum(deltas > 0)), "raw_deltas": [float(value) for value in deltas],
        }
    ap = report["average_precision"]
    conditions = {
        "ap": ap["mean_delta"] >= 0.002 or ap["wins"] >= 4,
        "roc_auc": report["roc_auc"]["mean_delta"] >= -0.001,
        "f1": report["f1"]["mean_delta"] >= -0.002,
    }
    return bool(all(conditions.values())), conditions, report


def run_stage(X, y, stage, baseline_configuration, candidates, seed=RANDOM_STATE):
    inner_frames, baseline_rows, candidate_rows, selections = [], [], [], []
    for outer_fold, (fit, validation) in enumerate(deterministic_splits(y, OUTER_FOLDS, seed), 1):
        X_fit, y_fit = X.iloc[fit], y[fit]
        selected, threshold, inner, _ = inner_oof_selection(
            X_fit, y_fit, candidates, outer_fold, stage, seed,
        )
        baseline_summary = inner[
            (inner.configuration_id == baseline_configuration.configuration_id)
            & (inner.row_type == "summary")
        ].iloc[0]
        baseline_threshold = float(baseline_summary.selected_threshold)
        inner_frames.append(inner)
        baseline_rows.append(evaluate_outer_configuration(
            X_fit, y_fit, X.iloc[validation], y[validation], baseline_configuration,
            baseline_threshold, stage, outer_fold, "baseline",
        ))
        candidate_rows.append(evaluate_outer_configuration(
            X_fit, y_fit, X.iloc[validation], y[validation], selected,
            threshold, stage, outer_fold, "candidate",
        ))
        selections.append({"outer_fold": outer_fold, "configuration_id": selected.configuration_id,
                           "threshold": threshold})
        print(f"V2 stage {stage}, outer fold {outer_fold}/{OUTER_FOLDS}: {selected.configuration_id}", flush=True)
    accepted, conditions, paired = stage_acceptance(baseline_rows, candidate_rows)
    full_selected, full_threshold, full_inner, _ = inner_oof_selection(
        X, y, candidates, 0, stage, seed,
    )
    full_inner["row_type"] = "full_training_" + full_inner["row_type"].astype(str)
    inner_frames.append(full_inner)
    next_configuration = full_selected if accepted else baseline_configuration
    if not accepted:
        baseline_full = full_inner[
            (full_inner.configuration_id == baseline_configuration.configuration_id)
            & (full_inner.row_type == "full_training_summary")
        ].iloc[0]
        full_threshold = float(baseline_full.selected_threshold)
    decision = {
        "stage": stage, "accepted": accepted, "conditions": conditions,
        "baseline_configuration_id": baseline_configuration.configuration_id,
        "candidate_family_winner_id": full_selected.configuration_id,
        "retained_configuration_id": next_configuration.configuration_id,
        "next_threshold": full_threshold, "paired_deltas": paired,
        "outer_fold_selections": selections,
    }
    return next_configuration, full_threshold, pd.concat(inner_frames, ignore_index=True), pd.DataFrame(
        baseline_rows + candidate_rows
    ), decision
