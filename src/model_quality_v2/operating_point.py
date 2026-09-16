"""Leakage-safe operating-point study for the frozen Model Quality V2 model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.config import PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR
from src.data.dataset import find_dataset, load_dataset, split_dataset
from src.evaluation.metrics import classification_metrics

from .config import V2Configuration
from .custom_v2_knn import CustomV2KNNClassifier
from .transforms import V2Preprocessor


OUTPUT_DIR = RESULTS_DIR / "model_quality_v2/operating_point"
FINAL_CONFIGURATION = RESULTS_DIR / "model_quality_v2/final_v2_configuration.json"
EXPECTED_CONFIGURATION = {
    "k": 101,
    "metric": "euclidean",
    "weights": "distance",
    "money_transform": "signed_log",
    "pay_representation": "structured",
    "feature_blocks": ["A", "B"],
    "pay_group_weight": 1.0,
}
POINT_NAMES = (
    "current_v2_threshold",
    "min_fp_recall_50",
    "min_fp_f1_52",
    "min_fp_accuracy_guard",
    "max_precision_recall_50",
    "f0_5_optimized",
)
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_RESAMPLES = 10_000
LEGACY_NOTICE = (
    "Legacy held-out test comparison; this test has prior analyst exposure and is "
    "not independent validation for operating-point selection."
)
CONSTRAINT_DEFINITIONS = {
    "current_v2_threshold": {
        "threshold": "already-frozen final V2 threshold; no reselection",
    },
    "min_fp_recall_50": {
        "objective": "minimize FP", "constraints": {"recall_min": 0.50},
        "tie_break": ["higher F1", "higher precision", "higher balanced accuracy", "larger threshold"],
    },
    "min_fp_f1_52": {
        "objective": "minimize FP", "constraints": {"f1_min": 0.52},
        "tie_break": ["higher recall", "higher precision", "higher balanced accuracy", "larger threshold"],
    },
    "min_fp_accuracy_guard": {
        "objective": "minimize FP",
        "constraints": {"accuracy_min": "reference OOF accuracy - 0.005", "recall_min": 0.48, "f1_min": 0.51},
        "tie_break": ["higher F1", "higher recall", "higher precision", "larger threshold"],
    },
    "max_precision_recall_50": {
        "objective": "maximize precision", "constraints": {"recall_min": 0.50},
        "tie_break": ["fewer FP", "higher F1", "higher balanced accuracy", "larger threshold"],
    },
    "f0_5_optimized": {
        "objective": "maximize F0.5",
        "tie_break": ["higher F1", "higher recall", "fewer FP", "threshold closest to current V2 threshold", "larger threshold"],
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values):
    values = np.asarray(values, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_at_threshold(scores, threshold):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector")
    return (scores >= float(threshold)).astype(np.int64)


def f_beta_from_counts(tp, fp, fn, beta=0.5):
    beta_squared = beta * beta
    numerator = (1.0 + beta_squared) * np.asarray(tp, dtype=float)
    denominator = numerator + beta_squared * np.asarray(fn, dtype=float) + np.asarray(fp, dtype=float)
    return np.divide(
        numerator, denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=float), where=denominator != 0,
    )


def threshold_frontier(y_true, scores, extra_thresholds=()):
    """Enumerate every attainable >= classification state in O(n log n)."""
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    thresholds = np.unique(np.r_[
        scores, 0.0, 0.5, 1.0, np.nextafter(1.0, np.inf),
        np.asarray(tuple(extra_thresholds), dtype=float),
    ])
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
    precision = divide(tp, tp + fp)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    return pd.DataFrame({
        "threshold": thresholds, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tp + tn) / len(labels), "precision": precision,
        "recall": recall, "specificity": specificity, "f1": f1,
        "f0_5": f_beta_from_counts(tp, fp, fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "predicted_positive_rate": (tp + fp) / len(labels),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, fn + tp),
    })


def _first_sorted(frame, columns, ascending, name):
    if frame.empty:
        raise ValueError(f"No threshold satisfies the predeclared constraints for {name}")
    return frame.sort_values(columns, ascending=ascending, kind="stable", na_position="last").iloc[0]


def select_predeclared_points(frontier, current_threshold):
    current_rows = frontier[frontier.threshold == current_threshold]
    if len(current_rows) != 1:
        raise ValueError("Frontier must contain the exact frozen current threshold once")
    reference = current_rows.iloc[0]
    points = {"current_v2_threshold": reference}
    feasible = frontier[frontier.recall >= 0.50]
    points["min_fp_recall_50"] = _first_sorted(
        feasible, ["FP", "f1", "precision", "balanced_accuracy", "threshold"],
        [True, False, False, False, False], "min_fp_recall_50",
    )
    feasible = frontier[frontier.f1 >= 0.52]
    points["min_fp_f1_52"] = _first_sorted(
        feasible, ["FP", "recall", "precision", "balanced_accuracy", "threshold"],
        [True, False, False, False, False], "min_fp_f1_52",
    )
    feasible = frontier[
        (frontier.accuracy >= float(reference.accuracy) - 0.005)
        & (frontier.recall >= 0.48) & (frontier.f1 >= 0.51)
    ]
    points["min_fp_accuracy_guard"] = _first_sorted(
        feasible, ["FP", "f1", "recall", "precision", "threshold"],
        [True, False, False, False, False], "min_fp_accuracy_guard",
    )
    feasible = frontier[(frontier.recall >= 0.50) & frontier.precision.notna()]
    points["max_precision_recall_50"] = _first_sorted(
        feasible, ["precision", "FP", "f1", "balanced_accuracy", "threshold"],
        [False, True, False, False, False], "max_precision_recall_50",
    )
    candidates = frontier.assign(distance_to_current=(frontier.threshold - current_threshold).abs())
    points["f0_5_optimized"] = _first_sorted(
        candidates, ["f0_5", "f1", "recall", "FP", "distance_to_current", "threshold"],
        [False, False, False, True, True, False], "f0_5_optimized",
    )
    return points


class _FenwickMaximum:
    def __init__(self, size):
        self.values = [None] * (size + 1)

    @staticmethod
    def _better(left, right):
        if left is None:
            return right
        if right is None:
            return left
        # Maximum F1; among equal F1 retain lower FP and higher recall.
        return max((left, right), key=lambda item: (item[0], -item[1], item[2]))

    def update(self, index, value):
        while index < len(self.values):
            self.values[index] = self._better(self.values[index], value)
            index += index & -index

    def query(self, index):
        result = None
        while index:
            result = self._better(result, self.values[index])
            index -= index & -index
        return result


def pareto_frontiers(frontier):
    """Return the 3D nondominated set and min-FP point for each recall level."""
    ordered = frontier.sort_values(
        ["FP", "recall", "f1", "threshold"],
        ascending=[True, False, False, False], kind="stable",
    ).copy()
    recalls = np.sort(ordered.recall.dropna().unique())[::-1]
    recall_rank = {value: index + 1 for index, value in enumerate(recalls)}
    tree = _FenwickMaximum(len(recalls))
    nondominated = []
    tolerance = 1e-15
    for row in ordered.itertuples():
        prior = tree.query(recall_rank[row.recall])
        dominated = False
        if prior is not None:
            prior_f1, prior_fp, prior_recall = prior
            dominated = (
                prior_f1 > row.f1 + tolerance
                or (abs(prior_f1 - row.f1) <= tolerance
                    and (prior_fp < row.FP or prior_recall > row.recall + tolerance))
            )
        nondominated.append(not dominated)
        tree.update(recall_rank[row.recall], (row.f1, row.FP, row.recall))
    ordered["pareto_3d"] = nondominated
    best_2d_indices = []
    for _, group in frontier.groupby("recall", dropna=False):
        best = group.sort_values(
            ["FP", "f1", "precision", "threshold"],
            ascending=[True, False, False, False], kind="stable", na_position="last",
        ).index[0]
        best_2d_indices.append(best)
    frontier_flags = pd.Series(False, index=frontier.index)
    frontier_flags.loc[best_2d_indices] = True
    ordered["frontier_2d"] = ordered.index.map(frontier_flags).astype(bool)
    result = ordered[ordered.pareto_3d | ordered.frontier_2d].copy()
    result["frontier_type"] = np.select(
        [result.pareto_3d & result.frontier_2d, result.pareto_3d],
        ["both", "fp_recall_f1_pareto"], default="min_fp_by_recall",
    )
    return result.sort_values(["FP", "recall", "threshold"], kind="stable"), int(ordered.pareto_3d.sum()), int(frontier_flags.sum())


def verify_final_v2_freeze(path=FINAL_CONFIGURATION):
    path = Path(path)
    final = read_json(path)
    configuration = final.get("configuration", {})
    actual = {key: configuration.get(key) for key in EXPECTED_CONFIGURATION}
    if actual != EXPECTED_CONFIGURATION:
        raise ValueError(f"Frozen V2 configuration mismatch: expected {EXPECTED_CONFIGURATION}, found {actual}")
    if not final.get("frozen") or final.get("legacy_test_evaluated") is not False:
        raise ValueError("Final V2 configuration is not in its required frozen state")
    for relative, expected in final.get("source_hashes", {}).items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"Frozen V2 source hash mismatch: {relative}")
    return final


def generate_oof(dataset, train_indices, configuration):
    X = dataset.X.iloc[train_indices].reset_index(drop=True)
    y = dataset.y[train_indices]
    scores = np.full(len(y), np.nan)
    folds = np.zeros(len(y), dtype=np.int64)
    visits = np.zeros(len(y), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    with threadpool_limits(limits=1):
        for fold, (fit, validation) in enumerate(splitter.split(X, y), 1):
            preprocessor = V2Preprocessor(configuration)
            X_fit = preprocessor.fit_transform(X.iloc[fit], y[fit])
            X_validation = preprocessor.transform(X.iloc[validation])
            model = KNeighborsClassifier(
                n_neighbors=configuration.k, metric="euclidean", weights="distance",
                algorithm="brute", n_jobs=1,
            ).fit(X_fit, y[fit])
            scores[validation] = model.predict_proba(X_validation)[:, 1]
            folds[validation] = fold
            visits[validation] += 1
            print(f"Operating-point OOF fold {fold}/5: {len(validation)} held-out rows", flush=True)
    if not np.all(visits == 1):
        raise ValueError("Every training row must be scored exactly once")
    return pd.DataFrame({
        "training_position": np.arange(len(y)),
        "row_id": dataset.row_ids[train_indices],
        "original_id": dataset.original_ids[train_indices],
        "fold": folds, "y_true": y, "probability_class_1": scores,
    })


def validate_oof(frame, dataset, train_indices):
    required = {"training_position", "row_id", "original_id", "fold", "y_true", "probability_class_1"}
    if set(frame.columns) != required or len(frame) != len(train_indices):
        raise ValueError("OOF shape or columns do not match the training partition")
    if not np.array_equal(frame.training_position.to_numpy(), np.arange(len(train_indices))):
        raise ValueError("OOF training positions are incomplete, duplicated, or out of order")
    if frame.row_id.duplicated().any() or frame.training_position.duplicated().any():
        raise ValueError("OOF row identifiers must be unique")
    if not np.array_equal(frame.row_id.to_numpy(), dataset.row_ids[train_indices]):
        raise ValueError("OOF rows are not exactly the fixed training partition")
    if not np.array_equal(frame.y_true.to_numpy(), dataset.y[train_indices]):
        raise ValueError("OOF labels are not aligned to the fixed training partition")
    if set(frame.fold.unique()) != {1, 2, 3, 4, 5} or frame.isna().any().any():
        raise ValueError("OOF folds or values are incomplete")
    scores = frame.probability_class_1.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("OOF probabilities are invalid")
    test_rows = set(dataset.row_ids).difference(dataset.row_ids[train_indices])
    if set(frame.row_id).intersection(test_rows):
        raise ValueError("OOF output contains legacy-test rows")


def _metrics_dict(row):
    names = (
        "threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall",
        "specificity", "f1", "f0_5", "balanced_accuracy", "predicted_positive_rate",
        "false_positive_rate", "false_negative_rate",
    )
    return {name: (int(row[name]) if name in {"TN", "FP", "FN", "TP"} else float(row[name])) for name in names}


def _selection_digest(value):
    stable = {key: item for key, item in value.items() if key not in {
        "freeze_sha256", "legacy_test_evaluated", "legacy_artifact_hashes",
    }}
    payload = json.dumps(_json_safe(stable), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _study_source_hashes():
    paths = (
        "src/model_quality_v2/operating_point.py",
        "src/model_quality_v2/reporting_operating_point.py",
        "src/model_quality_v2/config.py",
        "src/model_quality_v2/transforms.py",
        "src/model_quality_v2/features.py",
        "src/model_quality_v2/custom_v2_knn.py",
        "src/data/dataset.py",
    )
    return {name: sha256(PROJECT_ROOT / name) for name in paths}


def freeze_operating_points(final, source, oof, frontier, points, pareto_sizes, output_dir):
    value = {
        "frozen": True,
        "selection_source": "five-fold OOF scores from the original 24,000-row training partition only",
        "model_configuration": final["configuration"],
        "model_configuration_sha256": sha256(FINAL_CONFIGURATION),
        "current_v2_threshold": float(final["threshold"]),
        "dataset_sha256": sha256(source), "split_seed": RANDOM_STATE, "oof_folds": 5,
        "training_rows": len(oof), "threshold_candidate_count": len(frontier),
        "pareto_3d_size": pareto_sizes[0], "frontier_2d_size": pareto_sizes[1],
        "oof_ranking_metrics": {
            "roc_auc": float(roc_auc_score(oof.y_true, oof.probability_class_1)),
            "average_precision": float(average_precision_score(oof.y_true, oof.probability_class_1)),
        },
        "constraint_definitions": CONSTRAINT_DEFINITIONS,
        "operating_points": {name: _metrics_dict(points[name]) for name in POINT_NAMES},
        "artifact_hashes": {
            "oof_scores.csv": sha256(output_dir / "oof_scores.csv"),
            "threshold_frontier.csv": sha256(output_dir / "threshold_frontier.csv"),
            "threshold_pareto_frontier.csv": sha256(output_dir / "threshold_pareto_frontier.csv"),
        },
        "source_hashes": _study_source_hashes(),
        "legacy_test_evaluated": False,
    }
    value["freeze_sha256"] = _selection_digest(value)
    write_json(output_dir / "operating_points.json", value)
    return value


def guard_legacy_state(selection, verify_rerun=False):
    evaluated = selection.get("legacy_test_evaluated")
    if verify_rerun:
        if evaluated is not True:
            raise ValueError("Verification rerun requires a completed legacy evaluation")
    elif evaluated is not False:
        raise ValueError("Legacy evaluation already completed; use --verify-rerun explicitly")


def verify_operating_point_freeze(output_dir=OUTPUT_DIR, data=None, verify_rerun=False):
    output_dir = Path(output_dir)
    selection_path = output_dir / "operating_points.json"
    if not selection_path.is_file():
        raise ValueError("Frozen operating_points.json is missing")
    selection = read_json(selection_path)
    guard_legacy_state(selection, verify_rerun)
    final = verify_final_v2_freeze()
    if sha256(FINAL_CONFIGURATION) != selection["model_configuration_sha256"]:
        raise ValueError("Frozen V2 model configuration hash changed")
    if final["configuration"] != selection["model_configuration"]:
        raise ValueError("Frozen V2 model configuration changed")
    source = find_dataset(data)
    if sha256(source) != selection["dataset_sha256"]:
        raise ValueError("Dataset changed after operating-point freeze")
    for name, expected in selection["artifact_hashes"].items():
        if sha256(output_dir / name) != expected:
            raise ValueError(f"Frozen training artifact changed: {name}")
    for name, expected in selection["source_hashes"].items():
        if sha256(PROJECT_ROOT / name) != expected:
            raise ValueError(f"Operating-point source changed after freeze: {name}")
    if verify_rerun:
        legacy_hashes = selection.get("legacy_artifact_hashes")
        if not legacy_hashes:
            raise ValueError("Completed legacy artifact hashes are missing")
        for name, expected in legacy_hashes.items():
            if sha256(output_dir / name) != expected:
                raise ValueError(f"Completed legacy artifact changed: {name}")
    if _selection_digest(selection) != selection["freeze_sha256"]:
        raise ValueError("Frozen operating-point selection digest changed")
    frontier = pd.read_csv(output_dir / "threshold_frontier.csv", float_precision="round_trip")
    recomputed = select_predeclared_points(frontier, selection["current_v2_threshold"])
    for name in POINT_NAMES:
        if _metrics_dict(recomputed[name]) != selection["operating_points"][name]:
            raise ValueError(f"Frozen operating point no longer matches its rule: {name}")
    return selection, final, source


def tradeoff_against_reference(reference, candidate):
    fp_avoided = int(reference["FP"] - candidate["FP"])
    tp_lost = int(reference["TP"] - candidate["TP"])
    result = {
        "FP_avoided": fp_avoided,
        "additional_FN": int(candidate["FN"] - reference["FN"]),
        "TP_lost": tp_lost,
        "TN_gained": int(candidate["TN"] - reference["TN"]),
        "net_correct_classification_change": int(candidate["TN"] - reference["TN"] - tp_lost),
    }
    for metric in ("accuracy", "precision", "recall", "f1", "f0_5"):
        result[f"{metric}_delta"] = float(candidate[metric] - reference[metric])
    result["FP_avoided_per_TP_lost"] = float("inf") if tp_lost == 0 and fp_avoided > 0 else (
        float(fp_avoided / tp_lost) if tp_lost != 0 else float("nan")
    )
    result["TP_lost_per_100_FP_avoided"] = (
        float(100 * tp_lost / fp_avoided) if fp_avoided != 0 else float("nan")
    )
    return result


def _metrics_at_threshold(y, scores, threshold):
    predictions = predict_at_threshold(scores, threshold)
    values = classification_metrics(y, predictions)
    values.pop("test_samples")
    values["threshold"] = float(threshold)
    values["f0_5"] = float(f_beta_from_counts(values["TP"], values["FP"], values["FN"]))
    values["predicted_positive_rate"] = float(predictions.mean())
    return values, predictions


def manual_sklearn_parity(configuration, X_train, y_train, X_test, row_ids, original_ids,
                          points, sklearn_scores, manual_scores=None):
    if manual_scores is None:
        manual = CustomV2KNNClassifier(configuration.k).fit(X_train, y_train)
        manual_scores = manual.predict_proba(X_test)[:, 1]
    differences = np.abs(manual_scores - sklearn_scores)
    result = {
        "rows": len(X_test),
        "maximum_absolute_probability_difference": float(differences.max()),
        "probability_differences_gt_1e_10": int(np.sum(differences > 1e-10)),
        "probability_difference_rows": [],
        "operating_points": {},
    }
    for index in np.flatnonzero(differences > 1e-10):
        result["probability_difference_rows"].append({
            "test_position": int(index), "row_id": int(row_ids[index]),
            "original_id": str(original_ids[index]),
            "sklearn_probability": float(sklearn_scores[index]),
            "manual_probability": float(manual_scores[index]),
            "absolute_difference": float(differences[index]),
        })
    for name in POINT_NAMES:
        threshold = points[name]["threshold"]
        sklearn_labels = predict_at_threshold(sklearn_scores, threshold)
        manual_labels = predict_at_threshold(manual_scores, threshold)
        mismatch = np.flatnonzero(sklearn_labels != manual_labels)
        result["operating_points"][name] = {
            "threshold": threshold, "agreement_count": int(len(X_test) - len(mismatch)),
            "disagreement_count": int(len(mismatch)),
            "disagreements": [{
                "test_position": int(index), "row_id": int(row_ids[index]),
                "original_id": str(original_ids[index]),
                "sklearn_probability": float(sklearn_scores[index]),
                "manual_probability": float(manual_scores[index]),
            } for index in mismatch],
        }
    return result


def paired_bootstrap(y_true, predictions, reference_name="current_v2_threshold"):
    labels = np.asarray(y_true, dtype=np.int64)
    names = [name for name in POINT_NAMES if name != reference_name]
    metric_names = ("FP", "TP", "FN", "accuracy", "precision", "recall", "f1", "f0_5", "balanced_accuracy")
    observed = {}

    def batch_metrics(actual, predicted):
        tp = np.sum((actual == 1) & (predicted == 1), axis=1)
        fp = np.sum((actual == 0) & (predicted == 1), axis=1)
        fn = np.sum((actual == 1) & (predicted == 0), axis=1)
        tn = np.sum((actual == 0) & (predicted == 0), axis=1)

        def divide(a, b):
            return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b != 0)

        recall = divide(tp, tp + fn)
        specificity = divide(tn, tn + fp)
        return {
            "FP": fp, "TP": tp, "FN": fn, "accuracy": (tp + tn) / actual.shape[1],
            "precision": divide(tp, tp + fp), "recall": recall,
            "f1": divide(2 * tp, 2 * tp + fp + fn),
            "f0_5": f_beta_from_counts(tp, fp, fn),
            "balanced_accuracy": (recall + specificity) / 2,
        }

    single_actual = labels[None, :]
    single = {name: batch_metrics(single_actual, predictions[name][None, :]) for name in POINT_NAMES}
    for name in names:
        observed[name] = {metric: float(single[name][metric][0] - single[reference_name][metric][0]) for metric in metric_names}
    distributions = {(name, metric): np.empty(BOOTSTRAP_RESAMPLES) for name in names for metric in metric_names}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for start in range(0, BOOTSTRAP_RESAMPLES, 200):
        stop = min(start + 200, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        sampled_labels = labels[indices]
        reference = batch_metrics(sampled_labels, predictions[reference_name][indices])
        for name in names:
            candidate = batch_metrics(sampled_labels, predictions[name][indices])
            for metric in metric_names:
                distributions[name, metric][start:stop] = candidate[metric] - reference[metric]
    rows = []
    for name in names:
        for metric in metric_names:
            values = distributions[name, metric]
            finite = values[np.isfinite(values)]
            lower, upper = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            rows.append({
                "operating_point": name, "reference": reference_name, "metric": metric,
                "difference_definition": "candidate minus current V2 threshold",
                "observed_difference": observed[name][metric],
                "ci_lower_95": float(lower), "ci_upper_95": float(upper),
                "includes_zero": bool(lower <= 0 <= upper),
                "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            })
    return pd.DataFrame(rows)


def run_oof(data=None, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Operating-point output exists; refusing to overwrite frozen work")
    final = verify_final_v2_freeze()
    source = find_dataset(data)
    dataset = load_dataset(source)
    train, _ = split_dataset(dataset)
    configuration = V2Configuration.from_dict(final["configuration"])
    oof = generate_oof(dataset, train, configuration)
    validate_oof(oof, dataset, train)
    frontier = threshold_frontier(
        oof.y_true.to_numpy(), oof.probability_class_1.to_numpy(), [final["threshold"]],
    )
    points = select_predeclared_points(frontier, final["threshold"])
    pareto, pareto_3d_size, frontier_2d_size = pareto_frontiers(frontier)
    output_dir.mkdir(parents=True, exist_ok=False)
    oof.to_csv(output_dir / "oof_scores.csv", index=False, float_format="%.17g", lineterminator="\n")
    frontier.to_csv(output_dir / "threshold_frontier.csv", index=False, float_format="%.17g", lineterminator="\n")
    pareto.to_csv(output_dir / "threshold_pareto_frontier.csv", index=False, float_format="%.17g", lineterminator="\n")
    from .reporting_operating_point import create_oof_outputs
    create_oof_outputs(output_dir, frontier, points, final["threshold"])
    selection = freeze_operating_points(
        final, source, oof, frontier, points, (pareto_3d_size, frontier_2d_size), output_dir,
    )
    print(f"Frozen {len(points)} operating points from {len(frontier)} exact OOF thresholds", flush=True)
    return selection


def _legacy_results(dataset, train, test, final, selection):
    configuration = V2Configuration.from_dict(final["configuration"])
    preprocessor = V2Preprocessor(configuration)
    X_train = preprocessor.fit_transform(dataset.X.iloc[train], dataset.y[train])
    X_test = preprocessor.transform(dataset.X.iloc[test])
    with threadpool_limits(limits=1):
        model = KNeighborsClassifier(
            n_neighbors=configuration.k, metric="euclidean", weights="distance",
            algorithm="brute", n_jobs=1,
        ).fit(X_train, dataset.y[train])
        sklearn_scores = model.predict_proba(X_test)[:, 1]
    manual_model = CustomV2KNNClassifier(configuration.k).fit(X_train, dataset.y[train])
    scores = manual_model.predict_proba(X_test)[:, 1]
    point_definitions = selection["operating_points"]
    rows, predictions = [], {}
    for name in POINT_NAMES:
        metrics, labels = _metrics_at_threshold(dataset.y[test], scores, point_definitions[name]["threshold"])
        rows.append({"operating_point": name, "validation_context": LEGACY_NOTICE, **metrics})
        predictions[name] = labels
    reference = rows[0]
    for row in rows:
        row.update(tradeoff_against_reference(reference, row))
    comparison = pd.DataFrame(rows)
    shared = {
        "validation_context": LEGACY_NOTICE, "rows": len(test),
        "score_source": "CustomV2KNNClassifier deterministic original-index semantics",
        "roc_auc": float(roc_auc_score(dataset.y[test], scores)),
        "average_precision": float(average_precision_score(dataset.y[test], scores)),
        "score_vector_sha256": sha256_array(scores),
    }
    parity = manual_sklearn_parity(
        configuration, X_train, dataset.y[train], X_test,
        dataset.row_ids[test], dataset.original_ids[test], point_definitions,
        sklearn_scores, manual_scores=scores,
    )
    bootstrap = paired_bootstrap(dataset.y[test], predictions)
    return comparison, shared, parity, bootstrap


def evaluate_legacy(data=None, output_dir=OUTPUT_DIR, verify_rerun=False):
    output_dir = Path(output_dir)
    selection, final, source = verify_operating_point_freeze(output_dir, data, verify_rerun)
    dataset = load_dataset(source)
    train, test = split_dataset(dataset)
    comparison, shared, parity, bootstrap = _legacy_results(dataset, train, test, final, selection)
    if verify_rerun:
        expected_comparison = pd.read_csv(output_dir / "legacy_operating_point_comparison.csv", float_precision="round_trip")
        expected_bootstrap = pd.read_csv(output_dir / "legacy_operating_point_bootstrap.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(comparison, expected_comparison, check_exact=False, rtol=1e-14, atol=1e-14)
        pd.testing.assert_frame_equal(bootstrap, expected_bootstrap, check_exact=False, rtol=1e-14, atol=1e-14)
        if shared != read_json(output_dir / "legacy_shared_score_metrics.json"):
            raise ValueError("Shared legacy score metrics changed on verification rerun")
        if parity != read_json(output_dir / "manual_v2_all_rows_parity.json"):
            raise ValueError("Manual/sklearn parity changed on verification rerun")
        print("Operating-point verification rerun reproduced every legacy artifact", flush=True)
        return comparison
    legacy_paths = (
        "legacy_operating_point_comparison.csv", "legacy_operating_point_bootstrap.csv",
        "legacy_shared_score_metrics.json", "manual_v2_all_rows_parity.json", "operating_point_report.md",
    )
    if any((output_dir / name).exists() for name in legacy_paths):
        raise ValueError("Legacy artifacts already exist; refusing to overwrite them")
    comparison.to_csv(output_dir / "legacy_operating_point_comparison.csv", index=False,
                      float_format="%.17g", lineterminator="\n")
    bootstrap.to_csv(output_dir / "legacy_operating_point_bootstrap.csv", index=False,
                     float_format="%.17g", lineterminator="\n")
    write_json(output_dir / "legacy_shared_score_metrics.json", shared)
    write_json(output_dir / "manual_v2_all_rows_parity.json", parity)
    from .reporting_operating_point import create_final_report
    create_final_report(output_dir, selection, comparison, shared, parity, bootstrap)
    selection["legacy_test_evaluated"] = True
    selection["legacy_artifact_hashes"] = {
        name: sha256(output_dir / name) for name in legacy_paths
    }
    write_json(output_dir / "operating_points.json", selection)
    print("Legacy comparison completed from frozen operating points", flush=True)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="UCI credit-default XLS/XLSX/CSV")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oof-only", action="store_true", help="Generate and freeze training-only operating points")
    mode.add_argument("--evaluate-legacy", action="store_true", help="Run the guarded one-time legacy comparison")
    mode.add_argument("--verify-rerun", action="store_true", help="Recompute and verify completed legacy artifacts")
    args = parser.parse_args(argv)
    if args.oof_only:
        run_oof(args.data)
    else:
        evaluate_legacy(args.data, verify_rerun=args.verify_rerun)


if __name__ == "__main__":
    main()
