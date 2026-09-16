"""Separate training-OOF threshold selection for the already frozen KNN model."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.config import CV_FOLDS, PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR
from src.common.utils import read_json
from src.data.dataset import find_dataset, load_dataset, split_dataset
from src.evaluation.metrics import classification_metrics
from src.preprocessing.pipeline import make_preprocessor
from .enhanced_custom_knn import EnhancedCustomKNNClassifier

QUALITY_DIR = RESULTS_DIR / "model_quality"
FROZEN_MODEL = {"selected_k": 25, "metric": "euclidean", "weights": "distance"}
PRECISION_FLOOR = 0.55
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_RESAMPLES = 10_000
THRESHOLD_NAMES = (
    "default_threshold", "f1_optimized_threshold", "recall_oriented_threshold",
)
SCORE_SOURCES = (
    "src/custom_knn/classifier.py", "src/model_quality/enhanced_custom_knn.py",
    "src/preprocessing/pipeline.py", "src/data/dataset.py", "src/common/config.py",
    "src/model_quality/threshold.py", "src/evaluation/metrics.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def _write_csv(path, frame):
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={"original_id": str})


def _scores(values):
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional vector")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be probabilities in [0, 1]")
    return scores


def predict_at_threshold(scores, threshold):
    scores = _scores(scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, np.inf):
        raise ValueError("Threshold must be between 0 and the all-negative boundary")
    return (scores >= threshold).astype(np.int64)


def compute_metrics(y_true, predictions):
    values = classification_metrics(y_true, predictions)
    values.pop("test_samples")
    values["predicted_positive_rate"] = float(np.mean(np.asarray(predictions) == 1))
    values["false_positive_rate"] = 1.0 - values["specificity"]
    values["false_negative_rate"] = 1.0 - values["recall"]
    return values


def build_threshold_grid(y_true, scores):
    """Enumerate every distinct >= decision state, using sorted cumulative counts."""
    scores = _scores(scores)
    labels = np.asarray(y_true)
    if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be aligned binary values")
    thresholds = np.unique(np.r_[scores, 0.0, 0.5, 1.0, np.nextafter(1.0, np.inf)])
    order = np.argsort(scores, kind="stable")
    positive_prefix = np.r_[0, np.cumsum(labels[order], dtype=np.int64)]
    below = np.searchsorted(scores[order], thresholds, side="left")
    fn = positive_prefix[below]
    tn = below - fn
    tp = int(labels.sum()) - fn
    fp = len(labels) - int(labels.sum()) - tn

    def divide(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b != 0)

    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return pd.DataFrame({
        "threshold": thresholds, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": (tp + tn) / len(labels), "precision": divide(tp, tp + fp),
        "recall": recall, "specificity": specificity,
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "predicted_positive_rate": (tp + fp) / len(labels),
        "false_positive_rate": 1 - specificity, "false_negative_rate": 1 - recall,
    })


def select_thresholds(grid, precision_floor=PRECISION_FLOOR):
    if not np.isfinite(precision_floor) or not 0 <= precision_floor <= 1:
        raise ValueError("Precision floor must be in [0, 1]")
    candidates = grid.dropna(subset=["f1", "balanced_accuracy", "recall"]).copy()
    if candidates.empty or not grid.threshold.eq(0.5).any():
        raise ValueError("A valid threshold grid including 0.5 is required")
    candidates["distance_to_default"] = abs(candidates.threshold - 0.5)
    primary = candidates.sort_values(
        ["f1", "balanced_accuracy", "recall", "distance_to_default", "threshold"],
        ascending=[False, False, False, True, False], kind="stable",
    ).iloc[0]
    feasible = candidates[candidates.precision >= precision_floor]
    secondary = None
    if not feasible.empty:
        secondary = float(feasible.sort_values(
            ["recall", "f1", "balanced_accuracy", "threshold"],
            ascending=False, kind="stable",
        ).iloc[0].threshold)
    return {
        "default_threshold": 0.5,
        "f1_optimized_threshold": float(primary.threshold),
        "recall_oriented_threshold": secondary,
        "precision_floor": float(precision_floor),
    }


def validate_oof(frame, expected_rows, expected_row_ids=None):
    required = {"training_position", "row_id", "original_id", "fold", "y_true",
                "probability_class_1"}
    if not required.issubset(frame) or len(frame) != expected_rows:
        raise ValueError("OOF rows/columns do not match the training partition")
    if not np.array_equal(frame.training_position, np.arange(expected_rows)):
        raise ValueError("OOF positions must be complete, unique, and in training order")
    if frame.row_id.duplicated().any() or frame[list(required)].isna().any().any():
        raise ValueError("Missing or duplicate OOF row identifiers/values")
    if expected_row_ids is not None and not np.array_equal(frame.row_id, expected_row_ids):
        raise ValueError("OOF rows differ from the fixed training partition")
    if not frame.fold.isin(range(1, CV_FOLDS + 1)).all() or frame.fold.nunique() != CV_FOLDS:
        raise ValueError("OOF assignments must cover the five folds")
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError("OOF labels must be binary")
    _scores(frame.probability_class_1)


def generate_oof(dataset, train_indices):
    """The only features or labels used here belong to the training indices."""
    X = dataset.X.iloc[train_indices].copy()
    y = dataset.y[train_indices].copy()
    scores = np.full(len(y), np.nan)
    assignments = np.zeros(len(y), dtype=int)
    visits = np.zeros(len(y), dtype=int)
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    with threadpool_limits(limits=1):
        for fold, (fit, validation) in enumerate(folds.split(X, y), 1):
            transformer = make_preprocessor(X.iloc[fit])
            X_fit = transformer.fit_transform(X.iloc[fit])
            X_validation = transformer.transform(X.iloc[validation])
            model = KNeighborsClassifier(
                n_neighbors=25, metric="euclidean", weights="distance",
                algorithm="brute", n_jobs=1,
            ).fit(X_fit, y[fit])
            scores[validation] = model.predict_proba(X_validation)[:, 1]
            assignments[validation] = fold
            visits[validation] += 1
            print(f"Threshold OOF fold {fold}/{CV_FOLDS}: {len(validation)} training rows", flush=True)
    if not np.all(visits == 1):
        raise ValueError("Every training row must be held out exactly once")
    frame = pd.DataFrame({
        "training_position": np.arange(len(y)), "row_id": dataset.row_ids[train_indices],
        "original_id": dataset.original_ids[train_indices], "fold": assignments,
        "y_true": y, "probability_class_1": scores,
    })
    validate_oof(frame, len(y), dataset.row_ids[train_indices])
    return frame


def verify_model(quality_dir=QUALITY_DIR):
    quality_dir = Path(quality_dir)
    selection = read_json(quality_dir / "selected_quality_parameters.json")
    if any(selection.get(key) != value for key, value in FROZEN_MODEL.items()):
        raise ValueError("The threshold study requires frozen k=25/euclidean/distance")
    if selection.get("test_evaluated") is not True:
        raise ValueError("The selected-model evaluation must already be complete")
    if sha256(quality_dir / "cv_quality_grid.csv") != selection["cv_results_sha256"]:
        raise ValueError("Frozen CV artifact hash mismatch")
    manifest = read_json(quality_dir / "baseline_manifest.json")
    if sha256(PROJECT_ROOT / "src/custom_knn/classifier.py") != manifest["baseline_source_sha256"]:
        raise ValueError("Accepted V5.1 source hash mismatch")
    return selection


def paired_bootstrap(y_true, scores, selection, resamples=BOOTSTRAP_RESAMPLES,
                     seed=BOOTSTRAP_SEED):
    """Use identical sampled row indices for every operating-point comparison."""
    actual = np.asarray(y_true)
    predictions = {name: predict_at_threshold(scores, selection[name])
                   for name in THRESHOLD_NAMES if selection[name] is not None}
    observed = {name: compute_metrics(actual, pred) for name, pred in predictions.items()}
    metrics = ("f1", "recall", "precision", "balanced_accuracy", "TP", "FP", "FN")
    candidates = [name for name in predictions if name != "default_threshold"]
    differences = {(name, metric): np.empty(resamples)
                   for name in candidates for metric in metrics}
    rng = np.random.default_rng(seed)

    def batch_metrics(labels, pred):
        tp = np.sum((labels == 1) & (pred == 1), axis=1)
        fp = np.sum((labels == 0) & (pred == 1), axis=1)
        fn = np.sum((labels == 1) & (pred == 0), axis=1)
        tn = np.sum((labels == 0) & (pred == 0), axis=1)

        def divide(a, b):
            return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)

        return {"TP": tp, "FP": fp, "FN": fn,
                "f1": divide(2 * tp, 2 * tp + fp + fn),
                "recall": divide(tp, tp + fn), "precision": divide(tp, tp + fp),
                "balanced_accuracy": (divide(tp, tp + fn) + divide(tn, tn + fp)) / 2}

    for start in range(0, resamples, 200):
        stop = min(start + 200, resamples)
        indices = rng.integers(0, len(actual), size=(stop - start, len(actual)))
        labels = actual[indices]
        default = batch_metrics(labels, predictions["default_threshold"][indices])
        for name in candidates:
            candidate = batch_metrics(labels, predictions[name][indices])
            for metric in metrics:
                differences[name, metric][start:stop] = candidate[metric] - default[metric]
    rows = []
    for (name, metric), values in differences.items():
        finite = values[np.isfinite(values)]
        bounds = np.percentile(finite, [2.5, 97.5]) if len(finite) else [np.nan, np.nan]
        rows.append({
            "threshold_name": name, "metric": metric,
            "observed_delta": observed[name][metric] - observed["default_threshold"][metric],
            "bootstrap_95_percentile_lower": bounds[0],
            "bootstrap_95_percentile_upper": bounds[1],
            "resamples": resamples, "seed": seed, "valid_resamples": len(finite),
        })
    return pd.DataFrame(rows)


def _score_diagnostics(X_train, y_train, X_test, row_ids, original_ids,
                       enhanced_scores, sklearn_scores, model, selection):
    """Investigate score differences without changing either frozen decision rule."""
    score_delta = np.abs(enhanced_scores - sklearn_scores)
    affected = score_delta > 1e-10
    agreements = {}
    for name in THRESHOLD_NAMES:
        threshold = selection[name]
        if threshold is None:
            continue
        left = predict_at_threshold(enhanced_scores, threshold)
        right = predict_at_threshold(sklearn_scores, threshold)
        disagreements = np.flatnonzero(left != right)
        affected[disagreements] = True
        agreements[name] = {
            "threshold": threshold, "label_disagreements": len(disagreements),
            "row_ids": row_ids[disagreements].tolist(),
            "threshold_between_differing_scores": int(np.sum(
                (np.minimum(enhanced_scores, sklearn_scores) < threshold)
                & (threshold <= np.maximum(enhanced_scores, sklearn_scores))
            )),
        }
    positions = np.flatnonzero(affected)
    rows = []
    if len(positions):
        # Use the full query batch here: scikit's optimized kernel can produce
        # different roundoff when the query shape/layout is changed.
        sklearn_distances, sklearn_indices = model.kneighbors(X_test)
        manual = EnhancedCustomKNNClassifier(25).fit(X_train, y_train)
        exact_distances, exact_indices = manual.kneighbors(X_test[positions])
        for local, position in enumerate(positions):
            left_indices = exact_indices[local]
            right_indices = sklearn_indices[position]
            same_set = set(left_indices) == set(right_indices)
            exact_zero = exact_distances[local] == 0
            difference = X_train[right_indices] - X_test[position]
            right_exact = np.sqrt(np.einsum("ij,ij->i", difference, difference))
            distance_error = np.max(abs(right_exact - sklearn_distances[position]))
            if exact_zero.any():
                cause = "exact-zero neighbour(s); scikit Euclidean kernel roundoff affects inverse weights"
            elif not same_set:
                cause = "different boundary neighbours: scikit kernel ordering versus exact training-index ties"
            else:
                cause = "same neighbours; floating-point distance/weighted-vote roundoff"
            row = {
                "test_index": int(position), "row_id": int(row_ids[position]),
                "original_id": str(original_ids[position]),
                "enhanced_probability": enhanced_scores[position],
                "sklearn_probability": sklearn_scores[position],
                "absolute_score_difference": score_delta[position],
                "exact_zero_neighbours": int(exact_zero.sum()),
                "same_neighbour_set": same_set, "max_distance_roundoff": distance_error,
                "cause": cause,
            }
            for name, agreement in agreements.items():
                threshold = agreement["threshold"]
                row[f"{name}_enhanced_label"] = int(enhanced_scores[position] >= threshold)
                row[f"{name}_sklearn_label"] = int(sklearn_scores[position] >= threshold)
                row[f"{name}_distance_to_score_interval"] = max(
                    0.0, min(enhanced_scores[position], sklearn_scores[position]) - threshold,
                    threshold - max(enhanced_scores[position], sklearn_scores[position]),
                )
            rows.append(row)
    diagnostics = {
        "test_rows": len(enhanced_scores),
        "score_differences_above_1e_10": int(np.sum(score_delta > 1e-10)),
        "max_absolute_probability_difference": float(score_delta.max()),
        "threshold_agreement": agreements,
        "weka_exploratory_status": "not run; Weka native scores are not assumed equivalent",
        "cost_sensitive_analysis": "not run; no business costs supplied",
    }
    columns = ["test_index", "row_id", "original_id", "enhanced_probability",
               "sklearn_probability", "absolute_score_difference", "exact_zero_neighbours",
               "same_neighbour_set", "max_distance_roundoff", "cause"]
    return diagnostics, pd.DataFrame(rows, columns=None if rows else columns)


def evaluate_thresholds(data=None, quality_dir=QUALITY_DIR, output_dir=None,
                        verify_rerun=False):
    quality_dir = Path(quality_dir)
    output_dir = Path(output_dir) if output_dir else quality_dir / "threshold_study"
    source = find_dataset(data)
    selection = load_threshold_selection(output_dir, quality_dir, verify_rerun, source)
    # The complete freeze check above must succeed before parsing test features
    # or labels. Reuse the project's fixed stratified split; never reselect.
    dataset = load_dataset(source)
    train, test = split_dataset(dataset)
    oof = _read_csv(output_dir / "oof_scores.csv")
    validate_oof(oof, len(train), dataset.row_ids[train])
    if not np.array_equal(oof.y_true, dataset.y[train]):
        raise ValueError("Training labels differ from frozen OOF labels")
    with threadpool_limits(limits=1):
        transformer = make_preprocessor(dataset.X.iloc[train])
        X_train = np.asarray(transformer.fit_transform(dataset.X.iloc[train]), dtype=np.float64)
        X_test = np.asarray(transformer.transform(dataset.X.iloc[test]), dtype=np.float64)
        y_train, y_test = dataset.y[train], dataset.y[test]
        enhanced = EnhancedCustomKNNClassifier(25).fit(X_train, y_train)
        scores = enhanced.predict_proba(X_test)[:, 1]
        repeat = enhanced.predict_proba(X_test)[:, 1]
        if not np.array_equal(scores, repeat):
            raise ValueError("Enhanced Custom scores are not repeatable")
        sklearn = KNeighborsClassifier(
            n_neighbors=25, metric="euclidean", weights="distance",
            algorithm="brute", n_jobs=1,
        ).fit(X_train, y_train)
        sklearn_scores = sklearn.predict_proba(X_test)[:, 1]
        diagnostics, disagreements = _score_diagnostics(
            X_train, y_train, X_test, dataset.row_ids[test], dataset.original_ids[test],
            scores, sklearn_scores, sklearn, selection,
        )

    # Preserve the already-published selected-model default result. The new >=
    # rule intentionally differs from argmax only on exact probability=0.5.
    reference = _read_csv(quality_dir / "predictions_enhanced_custom.csv")
    for column, expected in (("row_id", dataset.row_ids[test]),
                             ("original_id", dataset.original_ids[test]), ("y_true", y_test)):
        if not np.array_equal(reference[column], expected):
            raise ValueError(f"Selected-model reference {column} alignment mismatch")
    if not np.allclose(scores, reference.probability_class_1, rtol=0, atol=1e-12):
        raise ValueError("Enhanced score vector differs from the existing selected-model result")
    default_predictions = predict_at_threshold(scores, 0.5)
    changed = default_predictions != reference.y_pred.to_numpy()
    if np.any(changed & (scores != 0.5)):
        raise ValueError("Threshold 0.5 changed existing labels away from an exact score tie")
    diagnostics.update({
        "enhanced_repeatable": True, "reference_default_label_differences": int(changed.sum()),
        "exact_half_score_count": int(np.sum(scores == 0.5)),
        "reference_score_max_absolute_difference": float(np.max(abs(scores - reference.probability_class_1))),
        "reference_score_hash": sha256(quality_dir / "predictions_enhanced_custom.csv"),
        "threshold_freeze_sha256": selection["freeze_sha256"],
    })
    comparison = pd.DataFrame([
        {"threshold_name": name, "threshold": selection[name],
         **compute_metrics(y_test, predict_at_threshold(scores, selection[name]))}
        for name in THRESHOLD_NAMES if selection[name] is not None
    ])
    bootstrap = paired_bootstrap(y_test, scores, selection)
    score_metrics = _score_metrics(y_test, scores)
    original = _read_csv(quality_dir / "test_quality_comparison.csv")
    baseline = original[original.implementation == "baseline_custom_v5_1"].iloc[0].to_dict()
    default = comparison[comparison.threshold_name == "default_threshold"].iloc[0]
    selected_reference = original[original.implementation == "enhanced_custom"].iloc[0]
    for metric in ("TN", "FP", "FN", "TP", "f1", "recall", "precision", "balanced_accuracy"):
        if not changed.any() and not np.isclose(default[metric], selected_reference[metric], rtol=0, atol=1e-14):
            raise ValueError(f"Default result no longer reproduces selected-model {metric}")
    _write_csv(output_dir / "test_threshold_comparison.csv", comparison)
    _write_csv(output_dir / "threshold_bootstrap.csv", bootstrap)
    _write_csv(output_dir / "threshold_score_diagnostics.csv", disagreements)
    _write_csv(output_dir / "test_scores.csv", pd.DataFrame({
        "row_id": dataset.row_ids[test], "original_id": dataset.original_ids[test],
        "y_true": y_test, "enhanced_probability": scores, "sklearn_probability": sklearn_scores,
    }))
    _write_json(output_dir / "test_score_metrics.json", score_metrics)
    _write_json(output_dir / "threshold_consistency.json", diagnostics)
    from .threshold_reporting import write_test_report
    write_test_report(output_dir, comparison, bootstrap, selection, baseline, score_metrics, diagnostics)
    # Recheck the original on-disk freeze immediately before the only permitted
    # state transition. A verification rerun leaves the selection bytes intact.
    load_threshold_selection(output_dir, quality_dir, verify_rerun, source)
    if not verify_rerun:
        selection["test_evaluated"] = True
        _write_json(output_dir / "threshold_selection.json", selection)
    print("Threshold test evaluation complete; frozen thresholds unchanged", flush=True)
    return comparison


def _selection_digest(selection):
    frozen = {key: value for key, value in selection.items()
              if key not in {"test_evaluated", "freeze_sha256"}}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_threshold_selection(study_dir, quality_dir=QUALITY_DIR, verify_rerun=False,
                             data_path=None):
    """Verify immutable thresholds, evidence, and state before loading test data."""
    study_dir, quality_dir = Path(study_dir), Path(quality_dir)
    selection = read_json(study_dir / "threshold_selection.json")
    expected_state = True if verify_rerun else False
    if selection.get("test_evaluated") is not expected_state:
        raise ValueError("First evaluation requires false; --verify-rerun requires true")
    if selection.get("freeze_sha256") != _selection_digest(selection):
        raise ValueError("Frozen threshold selection hash mismatch")
    verify_model(quality_dir)
    if selection["frozen_model"] != FROZEN_MODEL:
        raise ValueError("Frozen model changed")
    if sha256(quality_dir / "selected_quality_parameters.json") != selection["model_selection_sha256"]:
        raise ValueError("Selected-model artifact changed since threshold freeze")
    for relative, expected in selection["source_hashes"].items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"Score-producing source changed: {relative}")
    for package, expected in selection.get("package_versions", {}).items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Package version changed since OOF selection: {package}")
    for name, expected in selection.get("reference_artifact_hashes", {}).items():
        if sha256(quality_dir / name) != expected:
            raise ValueError(f"Existing model-quality reference changed: {name}")
    for name, expected in selection["artifact_hashes"].items():
        if sha256(study_dir / name) != expected:
            raise ValueError(f"Frozen OOF artifact changed: {name}")
    if data_path is not None and sha256(data_path) != selection["dataset_sha256"]:
        raise ValueError("Dataset differs from the OOF dataset")
    oof = _read_csv(study_dir / "oof_scores.csv")
    validate_oof(oof, selection["oof_rows"])
    recomputed = select_thresholds(build_threshold_grid(oof.y_true, oof.probability_class_1))
    if any(selection[key] != value for key, value in recomputed.items()):
        raise ValueError("Frozen thresholds do not match training-only OOF selection")
    return selection


def _score_metrics(y, scores):
    return {"roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores))}


def run_oof(data=None, quality_dir=QUALITY_DIR, output_dir=None):
    quality_dir = Path(quality_dir)
    output_dir = Path(output_dir) if output_dir else quality_dir / "threshold_study"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Refusing to overwrite existing threshold-study artifacts")
    verify_model(quality_dir)
    source = find_dataset(data)
    dataset_hash = sha256(source)
    dataset = load_dataset(source)
    train, _ = split_dataset(dataset)
    manifest = read_json(quality_dir / "baseline_manifest.json")
    if len(train) != manifest["training_rows"]:
        raise ValueError("Training row count differs from the recorded experiment")
    oof = generate_oof(dataset, train)
    grid = build_threshold_grid(oof.y_true, oof.probability_class_1)
    selected = select_thresholds(grid)
    rows = []
    for name in THRESHOLD_NAMES:
        if selected[name] is not None:
            rows.append({"threshold_name": name,
                         **grid[grid.threshold == selected[name]].iloc[0].to_dict()})
    selection = _json_safe({
        "frozen_model": FROZEN_MODEL, "selection_source": "training-only OOF",
        "decision_rule": "predict class 1 if probability_class_1 >= threshold",
        **selected,
        "primary_rule": "max F1; max balanced accuracy; max recall; closest to 0.5; larger threshold",
        "secondary_rule": "precision >= 0.55; max recall; max F1; max balanced accuracy; larger threshold",
        "secondary_status": "available" if selected["recall_oriented_threshold"] is not None
                            else "unavailable: no OOF threshold satisfies precision >= 0.55",
        "candidate_method": "exact unique OOF scores plus 0, 0.5, 1 and nextafter(1,+inf) all-negative boundary",
        "candidate_count": len(grid), "oof_rows": len(oof),
        "oof_class_counts": {str(c): int((oof.y_true == c).sum()) for c in (0, 1)},
        "oof_fold_counts": {str(f): int((oof.fold == f).sum()) for f in range(1, 6)},
        "oof_metrics": rows, "oof_score_metrics": _score_metrics(oof.y_true, oof.probability_class_1),
        "split_seed": RANDOM_STATE, "cv_folds": CV_FOLDS,
        "dataset_sha256": dataset_hash,
        "model_selection_sha256": sha256(quality_dir / "selected_quality_parameters.json"),
        "cv_artifact_sha256": sha256(quality_dir / "cv_quality_grid.csv"),
        "reference_artifact_hashes": {
            name: sha256(quality_dir / name)
            for name in ("test_quality_comparison.csv", "predictions_enhanced_custom.csv")
        },
        "source_hashes": {name: sha256(PROJECT_ROOT / name) for name in SCORE_SOURCES},
        "package_versions": {p: importlib.metadata.version(p)
                             for p in ("numpy", "pandas", "scikit-learn", "threadpoolctl")},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "test_evaluated": False,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "oof_scores.csv", oof)
    _write_csv(output_dir / "threshold_grid.csv", grid)
    from .threshold_reporting import plot_oof_figures, write_oof_report
    write_oof_report(output_dir, grid, selection)
    plot_oof_figures(output_dir, grid, selection)
    selection["artifact_hashes"] = {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    selection["freeze_sha256"] = _selection_digest(selection)
    _write_json(output_dir / "threshold_selection.json", selection)
    print(f"Frozen thresholds from {len(oof)} OOF rows / {len(grid)} candidates: {selected}", flush=True)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to the existing UCI XLS/XLSX/CSV")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oof-only", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    mode.add_argument("--verify-rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.oof_only:
        run_oof(args.data)
    else:
        evaluate_thresholds(args.data, verify_rerun=args.verify_rerun)


if __name__ == "__main__":
    main()
