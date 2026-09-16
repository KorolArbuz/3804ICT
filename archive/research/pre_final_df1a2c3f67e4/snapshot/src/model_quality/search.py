"""Training-only search over the deliberately small KNN quality grid."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.config import CV_FOLDS, K_VALUES, PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR
from src.common.utils import load_pair, read_json, write_json
from src.data.dataset import load_dataset, split_dataset
from src.evaluation.metrics import classification_metrics
from src.preprocessing.pipeline import make_preprocessor
from .reporting import plot_cv_figures, write_cv_report
from .selection import (
    QUALITY_METRICS,
    aggregate_cv,
    configuration_row,
    row_to_dict,
    select_configuration,
)

OUTPUT_DIR = RESULTS_DIR / "model_quality"
METRICS = ("euclidean", "manhattan")
WEIGHTS = ("uniform", "distance")
CONTEXT_NOTE = (
    "Existing test metrics are contextual only and are not used for new model selection."
)
SELECTION_RULE = (
    "maximum mean class-1 F1; within 1e-12 maximum mean balanced accuracy; "
    "then maximum mean recall; then smaller k; then Euclidean before Manhattan; "
    "then uniform before distance"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_status():
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "status", "--short"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, errors="replace",
    )
    return completed.stdout.splitlines() if completed.returncode == 0 else [
        f"git status unavailable: {completed.stderr.strip()}"
    ]


def _protected_hashes():
    paths = [RESULTS_DIR / "selected_parameters.json"]
    paths.extend(sorted(RESULTS_DIR.glob("*runtime*")))
    runtime_dir = RESULTS_DIR / "runtime_benchmarks"
    if runtime_dir.exists():
        paths.extend(sorted(path for path in runtime_dir.rglob("*") if path.is_file()))
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
        for path in paths if path.is_file()
    }


def create_baseline_manifest(dataset, train_indices, test_indices, output_dir):
    selected_path = RESULTS_DIR / "selected_parameters.json"
    selected = read_json(selected_path)
    if int(selected["selected_k"]) != 19:
        raise ValueError("Accepted baseline selection must remain k=19")

    metrics_path = RESULTS_DIR / "metrics_comparison.csv"
    metrics = pd.read_csv(metrics_path)
    custom = metrics[metrics.implementation == "custom"]
    if len(custom) != 1:
        raise ValueError("Authoritative metrics must contain one custom baseline row")

    processed_dimensions = None
    train_npz = PROJECT_ROOT / "data/processed/train.npz"
    test_npz = PROJECT_ROOT / "data/processed/test.npz"
    if train_npz.is_file() and test_npz.is_file():
        processed_train, processed_test = load_pair(train_npz, test_npz)
        if not np.array_equal(processed_train["row_ids"], dataset.row_ids[train_indices]):
            raise ValueError("Processed training rows differ from the fixed split")
        if not np.array_equal(processed_test["row_ids"], dataset.row_ids[test_indices]):
            raise ValueError("Processed test rows differ from the fixed split")
        processed_dimensions = {
            "train": list(processed_train["X"].shape),
            "test": list(processed_test["X"].shape),
        }

    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)

    row = custom.iloc[0]
    metric_names = [
        "accuracy", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "roc_auc", "average_precision", "TN", "FP", "FN", "TP",
    ]
    manifest = {
        "baseline_source": "src/custom_knn/classifier.py",
        "baseline_source_sha256": _sha256(PROJECT_ROOT / "src/custom_knn/classifier.py"),
        "baseline_k": 19,
        "baseline_metric": "euclidean",
        "baseline_weights": "uniform",
        "split_seed": RANDOM_STATE,
        "test_fraction": 0.20,
        "cv_folds": CV_FOLDS,
        "k_candidates": list(K_VALUES),
        "distance_candidates": list(METRICS),
        "weight_candidates": list(WEIGHTS),
        "dataset_rows": len(dataset.y),
        "training_rows": len(train_indices),
        "test_rows": len(test_indices),
        "original_predictor_count": dataset.X.shape[1],
        "processed_dimensions": processed_dimensions,
        "package_versions": versions,
        "existing_baseline_test_metrics": {
            name: (int(row[name]) if name in {"TN", "FP", "FN", "TP"} else float(row[name]))
            for name in metric_names
        },
        "test_metric_context": CONTEXT_NOTE,
        "selected_parameters_sha256": _sha256(selected_path),
        "protected_artifact_hashes_before_experiment": _protected_hashes(),
        "git_status_before_experiment": _git_status(),
    }
    write_json(output_dir / "baseline_manifest.json", manifest)
    return manifest


def run_training_search(dataset, train_indices):
    X_train = dataset.X.iloc[train_indices].reset_index(drop=True)
    y_train = dataset.y[train_indices]
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    with threadpool_limits(limits=1):
        for fold, (fit_indices, validation_indices) in enumerate(
            folds.split(X_train, y_train), start=1
        ):
            transformer = make_preprocessor(X_train)
            X_fit = np.asarray(transformer.fit_transform(X_train.iloc[fit_indices]), dtype=np.float64)
            X_validation = np.asarray(transformer.transform(X_train.iloc[validation_indices]), dtype=np.float64)
            y_fit = y_train[fit_indices]
            y_validation = y_train[validation_indices]

            for metric in METRICS:
                for weights in WEIGHTS:
                    for k in K_VALUES:
                        model = KNeighborsClassifier(
                            n_neighbors=k, metric=metric, weights=weights,
                            algorithm="brute", n_jobs=1,
                        )
                        model.fit(X_fit, y_fit)
                        probabilities = model.predict_proba(X_validation)
                        positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
                        predictions = model.classes_[np.argmax(probabilities, axis=1)]
                        values = classification_metrics(
                            y_validation, predictions, probabilities[:, positive_index]
                        )
                        rows.append({
                            "fold": fold, "k": k, "metric": metric, "weights": weights,
                            **{name: values[name] for name in QUALITY_METRICS},
                            **{name: values[name] for name in ("TN", "FP", "FN", "TP")},
                        })
            print(f"Model-quality CV fold {fold}/{CV_FOLDS}: 64 configurations", flush=True)
    return pd.DataFrame(rows)


def search(data=None, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    selection_path = output_dir / "selected_quality_parameters.json"
    if selection_path.exists() and read_json(selection_path).get("test_evaluated") is True:
        raise RuntimeError("Refusing to overwrite a selection that has already been test-evaluated")

    dataset = load_dataset(data)
    train_indices, test_indices = split_dataset(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_baseline_manifest(dataset, train_indices, test_indices, output_dir)

    results = run_training_search(dataset, train_indices)
    if len(results) != CV_FOLDS * len(K_VALUES) * len(METRICS) * len(WEIGHTS):
        raise RuntimeError("Training search did not produce all 320 fold rows")
    results_path = output_dir / "cv_quality_grid.csv"
    results.to_csv(results_path, index=False, float_format="%.17g")

    summary = aggregate_cv(results)
    summary_path = output_dir / "cv_quality_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.17g")
    winner = select_configuration(summary)
    baseline = configuration_row(summary, 19, "euclidean", "uniform")
    delta_metrics = {
        name: float(winner[f"{name}_mean"] - baseline[f"{name}_mean"])
        for name in QUALITY_METRICS
    }
    selection = {
        "selected_k": int(winner.k),
        "metric": str(winner.metric),
        "weights": str(winner.weights),
        "selection_metric": "mean class-1 F1",
        "selection_rule": SELECTION_RULE,
        "tie_tolerance": 1e-12,
        "test_evaluated": False,
        "cv_results_sha256": _sha256(results_path),
        "baseline_cv_row": row_to_dict(baseline),
        "selected_candidate_cv_row": row_to_dict(winner),
        "selected_minus_baseline_mean_metric_deltas": delta_metrics,
        "test_metric_context": CONTEXT_NOTE,
    }
    write_json(selection_path, selection)
    write_cv_report(output_dir / "cv_quality_report.md", baseline, winner, summary)
    plot_cv_figures(summary, winner, output_dir)
    print(
        f"Frozen training-CV winner: k={int(winner.k)}, {winner.metric}, "
        f"{winner.weights}; test_evaluated=false -> {selection_path}", flush=True,
    )
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="UCI XLS/XLSX/CSV path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    search(args.data, args.output_dir)


if __name__ == "__main__":
    main()
