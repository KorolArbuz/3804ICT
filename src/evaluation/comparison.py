from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair, read_json

from .metrics import evaluate_file, validate_predictions

IMPLEMENTATIONS = ("custom", "sklearn", "weka")
DISPLAY = {"custom": "Custom Python KNN", "sklearn": "scikit-learn KNN", "weka": "Weka IBk"}


def compare_predictions(frames, implementations):
    agreements = []
    for left_name, right_name in combinations(implementations, 2):
        left_probabilities = frames[left_name].probability_class_1.to_numpy()
        right_probabilities = frames[right_name].probability_class_1.to_numpy()
        predictions_differ = (
            frames[left_name].y_pred.to_numpy()
            != frames[right_name].y_pred.to_numpy()
        )

        if np.std(left_probabilities) > 0 and np.std(right_probabilities) > 0:
            correlation = float(
                np.corrcoef(left_probabilities, right_probabilities)[0, 1]
            )
        else:
            correlation = float("nan")

        agreements.append(
            {
                "implementation_a": left_name,
                "implementation_b": right_name,
                "test_samples": len(left_probabilities),
                "class_agreement": float((~predictions_differ).mean()),
                "disagreement_count": int(predictions_differ.sum()),
                "probability_correlation": correlation,
                "max_absolute_probability_difference": float(
                    np.abs(left_probabilities - right_probabilities).max()
                ),
            }
        )
    return agreements


def compare(results_dir=RESULTS_DIR, train=PROCESSED_DATA_DIR / "train.npz",
            test=PROCESSED_DATA_DIR / "test.npz", implementations=None,
            selected_parameters=None, prediction_paths=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_parameters = Path(selected_parameters or results_dir / "selected_parameters.json")
    _, testing = load_pair(train, test)
    k = int(read_json(selected_parameters)["selected_k"])
    if k < 1:
        raise ValueError("Selected k must be at least 1")

    if implementations is None:
        implementations = []
        for name in IMPLEMENTATIONS:
            prediction_path = results_dir / f"predictions_{name}.csv"
            if prediction_path.exists():
                implementations.append(name)

    invalid_implementations = set(implementations) - set(IMPLEMENTATIONS)
    has_duplicates = len(set(implementations)) != len(implementations)
    if not implementations or has_duplicates or invalid_implementations:
        raise ValueError("Choose available distinct custom, sklearn and/or weka prediction files")

    frames, metrics = {}, []
    for name in implementations:
        if prediction_paths and name in prediction_paths:
            path = Path(prediction_paths[name])
        else:
            path = results_dir / f"predictions_{name}.csv"

        frame = validate_predictions(path, testing)
        if not frame.selected_k.eq(k).all():
            raise ValueError(f"{name} uses a different k from the CV selection")

        runtime_path = path.with_suffix(".runtime.json")
        runtime = read_json(runtime_path)
        if name == "weka":
            invalid_weka_configuration = (
                runtime.get("distance_normalization") is not False
                or runtime.get("internal_cross_validation") is not False
                or runtime.get("distance_weighting") != "none"
            )
            if invalid_weka_configuration:
                raise ValueError(
                    "Weka distance/CV/weighting does not meet the comparison contract"
                )

        frames[name] = frame
        metrics.append(evaluate_file(path, test, name))

    table = pd.DataFrame(metrics)
    table.to_csv(results_dir / "metrics_comparison.csv", index=False, na_rep="undefined")

    runtime_columns = [
        "implementation",
        "fit_seconds",
        "prediction_seconds",
        "milliseconds_per_sample",
    ]
    table[runtime_columns].to_csv(results_dir / "runtime_comparison.csv", index=False)

    confusion_columns = ["implementation", "TN", "FP", "FN", "TP"]
    table[confusion_columns].to_csv(
        results_dir / "confusion_matrices.csv",
        index=False,
    )

    agreements = compare_predictions(frames, implementations)
    agreement_columns = [
        "implementation_a",
        "implementation_b",
        "test_samples",
        "class_agreement",
        "disagreement_count",
        "probability_correlation",
        "max_absolute_probability_difference",
    ]
    agreement_table = pd.DataFrame(agreements, columns=agreement_columns)
    agreement_table.to_csv(
        results_dir / "prediction_agreement.csv",
        index=False,
        na_rep="undefined",
    )

    disagreements = sum(item["disagreement_count"] for item in agreements)
    print(
        f"Compared {len(metrics)} implementations; "
        f"{disagreements} pairwise class disagreements",
        flush=True,
    )
    return table
