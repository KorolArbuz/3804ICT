from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.config import (
    CV_FOLDS,
    K_VALUES,
    RANDOM_STATE,
    RESULTS_DIR,
    TIE_TOLERANCE,
)
from src.common.utils import write_json
from src.data.dataset import load_dataset, split_dataset
from src.evaluation.metrics import classification_metrics
from src.preprocessing.pipeline import make_preprocessor

CV_METRICS = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]


def run_cross_validation(dataset):
    train_indices, _ = split_dataset(dataset)
    X = dataset.X.iloc[train_indices]
    y = dataset.y[train_indices]

    class_counts = np.bincount(y)
    if min(class_counts) < CV_FOLDS:
        raise ValueError("Each training class needs at least five samples for stratified CV")

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []
    with threadpool_limits(limits=1):
        for fold_number, (fit_indices, validation_indices) in enumerate(
            cross_validation.split(X, y),
            start=1,
        ):
            if len(fit_indices) < max(K_VALUES):
                raise ValueError("Each CV training fold needs at least 31 samples")

            # Each fold fits its own preprocessor to avoid validation leakage.
            transformer = make_preprocessor(X)
            training_data = transformer.fit_transform(X.iloc[fit_indices])
            validation_data = transformer.transform(X.iloc[validation_indices])

            for k in K_VALUES:
                classifier = KNeighborsClassifier(
                    n_neighbors=k,
                    weights="uniform",
                    algorithm="brute",
                    metric="euclidean",
                    n_jobs=1,
                )
                classifier.fit(training_data, y[fit_indices])
                predictions = classifier.predict(validation_data)
                metrics = classification_metrics(y[validation_indices], predictions)

                row = {"fold": fold_number, "k": k}
                for metric_name in CV_METRICS:
                    row[metric_name] = metrics[metric_name]
                rows.append(row)

            print(
                f"CV fold {fold_number}/{CV_FOLDS}: evaluated all "
                f"{len(K_VALUES)} candidate k values",
                flush=True,
            )

    return pd.DataFrame(rows)


def aggregate_results(results):
    grouped = results.groupby("k")[CV_METRICS]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    return summary.reset_index()


def select_k(summary):
    best_f1 = summary.f1_mean.max()
    candidates = summary[summary.f1_mean >= best_f1 - TIE_TOLERANCE]

    best_balanced_accuracy = candidates.balanced_accuracy_mean.max()
    candidates = candidates[
        candidates.balanced_accuracy_mean >= best_balanced_accuracy - TIE_TOLERANCE
    ]
    selected_k = int(candidates.k.min())
    return selected_k


def tune(data=None, output=RESULTS_DIR / "cv_results.csv",
         selected_output=RESULTS_DIR / "selected_parameters.json"):
    dataset = load_dataset(data)
    results = run_cross_validation(dataset)
    summary = aggregate_results(results)
    selected_k = select_k(summary)

    output = Path(output)
    selected_output = Path(selected_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, na_rep="undefined")

    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, na_rep="undefined")

    selected = {
        "selected_k": selected_k,
        "selection": (
            "max mean F1; within 1e-12 max mean balanced accuracy; "
            "within 1e-12 smallest k"
        ),
        "test_evaluated": False,
    }
    write_json(selected_output, selected)

    from src.reporting.plots import plot_cv

    plot_cv(summary, selected_k, output.with_name("cv_f1_vs_k.png"))
    print(f"Selected k={selected_k} from training-only CV -> {selected_output}", flush=True)
    return selected
