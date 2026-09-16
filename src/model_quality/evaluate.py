"""One-time final test evaluation of the training-CV-frozen KNN candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.config import PROJECT_ROOT, RESULTS_DIR
from src.common.utils import read_json, write_json
from src.custom_knn.classifier import CustomKNNClassifier
from src.data.dataset import load_dataset, split_dataset
from src.evaluation.metrics import classification_metrics
from src.preprocessing.export import arff_text
from src.preprocessing.pipeline import make_preprocessor, validate_processed_feature_names
from src.weka_bridge.runner import run as run_weka
from .enhanced_custom_knn import EnhancedCustomKNNClassifier
from .reporting import plot_final_figures
from .selection import QUALITY_METRICS, require_frozen_selection

OUTPUT_DIR = RESULTS_DIR / "model_quality"
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_RESAMPLES = 10_000


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pretest_freeze(output_dir, selection, verify_rerun=False):
    expected_state = True if verify_rerun else False
    if selection["test_evaluated"] is not expected_state:
        raise RuntimeError(
            "Final evaluation requires test_evaluated=false; --verify-rerun requires "
            "an already completed artifact with test_evaluated=true"
        )
    grid_path = output_dir / "cv_quality_grid.csv"
    if _sha256(grid_path) != selection["cv_results_sha256"]:
        raise RuntimeError("Training CV results changed after parameter selection")

    manifest = read_json(output_dir / "baseline_manifest.json")
    classifier_path = PROJECT_ROOT / manifest["baseline_source"]
    if _sha256(classifier_path) != manifest["baseline_source_sha256"]:
        raise RuntimeError("Accepted Custom V5.1 source changed after baseline recording")
    selected_path = RESULTS_DIR / "selected_parameters.json"
    if _sha256(selected_path) != manifest["selected_parameters_sha256"]:
        raise RuntimeError("Accepted baseline selected_parameters.json changed")
    for relative_path, expected_hash in manifest[
        "protected_artifact_hashes_before_experiment"
    ].items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"Protected runtime/baseline artifact changed: {relative_path}")

    expected = (
        int(selection["selected_k"]),
        selection["metric"],
        selection["weights"],
    )
    implemented = (
        int(selection["selected_k"]),
        EnhancedCustomKNNClassifier.metric,
        EnhancedCustomKNNClassifier.weights,
    )
    if implemented != expected:
        raise RuntimeError(
            f"Enhanced Custom implements {implemented}, but CV froze {expected}"
        )


def _prediction_frame(dataset, test_indices, predictions, probabilities, k, metric, weights):
    return pd.DataFrame({
        "test_index": np.arange(len(test_indices)),
        "row_id": dataset.row_ids[test_indices],
        "original_id": dataset.original_ids[test_indices],
        "y_true": dataset.y[test_indices],
        "y_pred": np.asarray(predictions, dtype=np.int64),
        "probability_class_1": np.asarray(probabilities, dtype=np.float64),
        "selected_k": int(k),
        "metric": metric,
        "weights": weights,
    })


def _evaluate_model(name, configuration, k, metric, weights, model, X_train, y_train,
                    X_test, y_test, dataset, test_indices, output_path):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    positive_probabilities = probabilities[:, positive_index]
    frame = _prediction_frame(
        dataset, test_indices, predictions, positive_probabilities,
        k, metric, weights,
    )
    frame.to_csv(output_path, index=False, float_format="%.17g")
    return frame, {
        "implementation": name,
        "configuration": configuration,
        "k": int(k),
        "metric": metric,
        "weights": weights,
        **classification_metrics(y_test, predictions, positive_probabilities),
    }


def _write_weka_inputs(directory, X_train, y_train, X_test, y_test,
                       feature_names, dataset, test_indices):
    directory = Path(directory)
    train_path = directory / "train.arff"
    test_path = directory / "test.arff"
    train_path.write_text(arff_text(X_train, y_train, feature_names), encoding="utf-8", newline="\n")
    test_path.write_text(arff_text(X_test, y_test, feature_names), encoding="utf-8", newline="\n")
    pd.DataFrame({
        "row_id": dataset.row_ids[test_indices],
        "original_id": dataset.original_ids[test_indices],
    }).to_csv(directory / "test_ids.csv", index=False)
    return train_path, test_path


def _paired_bootstrap(y_true, baseline_predictions, candidate_predictions):
    actual = np.asarray(y_true, dtype=np.int8)
    baseline = np.asarray(baseline_predictions, dtype=np.int8)
    candidate = np.asarray(candidate_predictions, dtype=np.int8)
    random = np.random.RandomState(BOOTSTRAP_SEED)
    samples = {name: np.empty(BOOTSTRAP_RESAMPLES) for name in (
        "f1", "recall", "precision", "balanced_accuracy"
    )}

    def values(sample_actual, sample_predictions):
        tp = ((sample_actual == 1) & (sample_predictions == 1)).sum(axis=1)
        fp = ((sample_actual == 0) & (sample_predictions == 1)).sum(axis=1)
        fn = ((sample_actual == 1) & (sample_predictions == 0)).sum(axis=1)
        tn = ((sample_actual == 0) & (sample_predictions == 0)).sum(axis=1)
        precision = np.divide(tp, tp + fp, out=np.full(tp.shape, np.nan), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.full(tp.shape, np.nan), where=(tp + fn) != 0)
        specificity = np.divide(tn, tn + fp, out=np.full(tn.shape, np.nan), where=(tn + fp) != 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn,
                       out=np.full(tp.shape, np.nan), where=(2 * tp + fp + fn) != 0)
        return {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "balanced_accuracy": (recall + specificity) / 2,
        }

    batch_size = 200
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        indices = random.randint(0, len(actual), size=(stop - start, len(actual)))
        sampled_actual = actual[indices]
        base_values = values(sampled_actual, baseline[indices])
        candidate_values = values(sampled_actual, candidate[indices])
        for metric in samples:
            samples[metric][start:stop] = candidate_values[metric] - base_values[metric]

    baseline_observed = classification_metrics(actual, baseline)
    candidate_observed = classification_metrics(actual, candidate)
    rows = []
    for metric, differences in samples.items():
        rows.append({
            "metric": metric,
            "observed_delta": candidate_observed[metric] - baseline_observed[metric],
            "bootstrap_95_percentile_lower": np.nanpercentile(differences, 2.5),
            "bootstrap_95_percentile_upper": np.nanpercentile(differences, 97.5),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        })
    return pd.DataFrame(rows)


def _investigate_probability_differences(X_train, y_train, X_test, dataset,
                                         test_indices, sklearn_frame,
                                         enhanced_frame, k):
    differences = np.abs(
        enhanced_frame.probability_class_1.to_numpy()
        - sklearn_frame.probability_class_1.to_numpy()
    )
    affected = np.flatnonzero(differences > 1e-10)
    columns = [
        "test_index", "row_id", "scikit_probability", "enhanced_probability",
        "absolute_difference", "exact_zero_training_rows", "exact_zero_labels",
        "scikit_min_distance", "enhanced_min_distance", "same_neighbor_set", "cause",
    ]
    if not len(affected):
        return pd.DataFrame(columns=columns)

    sklearn_model = KNeighborsClassifier(
        n_neighbors=k, metric="euclidean", weights="distance",
        algorithm="brute", n_jobs=1,
    ).fit(X_train, y_train)
    enhanced_model = EnhancedCustomKNNClassifier(k).fit(X_train, y_train)
    sklearn_distances, sklearn_indices = sklearn_model.kneighbors(X_test[affected])
    enhanced_distances, enhanced_indices = enhanced_model.kneighbors(X_test[affected])
    rows = []
    for position, test_index in enumerate(affected):
        query = X_test[test_index]
        exact_squared = np.einsum("ij,ij->i", X_train - query, X_train - query)
        zero_indices = np.flatnonzero(exact_squared == 0.0)
        same_neighbours = set(sklearn_indices[position]) == set(enhanced_indices[position])
        if len(zero_indices):
            cause = "exact-zero distance represented as tiny positive by scikit kernel"
        elif not same_neighbours:
            cause = "equal/near-equal k-boundary distance resolved differently"
        else:
            cause = "pairwise-distance floating-point differences within same neighbour set"
        rows.append({
            "test_index": int(test_index),
            "row_id": int(dataset.row_ids[test_indices][test_index]),
            "scikit_probability": float(
                sklearn_frame.probability_class_1.iloc[test_index]
            ),
            "enhanced_probability": float(
                enhanced_frame.probability_class_1.iloc[test_index]
            ),
            "absolute_difference": float(differences[test_index]),
            "exact_zero_training_rows": int(len(zero_indices)),
            "exact_zero_labels": "".join(map(str, y_train[zero_indices].tolist())),
            "scikit_min_distance": float(sklearn_distances[position].min()),
            "enhanced_min_distance": float(enhanced_distances[position].min()),
            "same_neighbor_set": bool(same_neighbours),
            "cause": cause,
        })
    return pd.DataFrame(rows, columns=columns)


def _write_final_report(path, comparison, bootstrap, majority_accuracy, agreement, weka_status):
    baseline = comparison[comparison.implementation == "baseline_custom_v5_1"].iloc[0]
    selected = comparison[comparison.implementation == "enhanced_custom"].iloc[0]
    deltas = {name: selected[name] - baseline[name] for name in (
        "f1", "recall", "precision", "balanced_accuracy", "average_precision", "roc_auc",
        "TP", "FN", "FP",
    )}
    intervals = bootstrap.set_index("metric")
    text = f"""# Final model-quality analysis

## Frozen selection and test results

Training-only cross-validation selected **k={int(selected.k)}, {selected.metric},
{selected.weights} voting** before this test evaluation. The untouched test set was
then evaluated once with the accepted baseline and the frozen candidate.

| Metric | Baseline Custom V5.1 | Enhanced Custom | Delta |
|---|---:|---:|---:|
| Class-1 F1 | {baseline.f1:.4f} | {selected.f1:.4f} | {deltas['f1']:+.4f} |
| Recall | {baseline.recall:.4f} | {selected.recall:.4f} | {deltas['recall']:+.4f} |
| Precision | {baseline.precision:.4f} | {selected.precision:.4f} | {deltas['precision']:+.4f} |
| Balanced accuracy | {baseline.balanced_accuracy:.4f} | {selected.balanced_accuracy:.4f} | {deltas['balanced_accuracy']:+.4f} |
| Average precision | {baseline.average_precision:.4f} | {selected.average_precision:.4f} | {deltas['average_precision']:+.4f} |
| ROC-AUC | {baseline.roc_auc:.4f} | {selected.roc_auc:.4f} | {deltas['roc_auc']:+.4f} |

The confusion counts changed by **TP {deltas['TP']:+.0f}**, **FN {deltas['FN']:+.0f}**,
and **FP {deltas['FP']:+.0f}**. This states the measured trade-off directly; accuracy
alone is not treated as evidence of improved default detection.

The arithmetic majority-class reference accuracy is **{majority_accuracy:.4f}**.
This is an arithmetic majority-class reference, not a trained model.

## Paired bootstrap description

The deterministic paired bootstrap resampled the same test rows 10,000 times. Its
95% percentile intervals for candidate-minus-baseline differences were:

| Metric | Observed delta | 95% percentile interval |
|---|---:|---:|
| F1 | {intervals.loc['f1', 'observed_delta']:+.4f} | [{intervals.loc['f1', 'bootstrap_95_percentile_lower']:+.4f}, {intervals.loc['f1', 'bootstrap_95_percentile_upper']:+.4f}] |
| Recall | {intervals.loc['recall', 'observed_delta']:+.4f} | [{intervals.loc['recall', 'bootstrap_95_percentile_lower']:+.4f}, {intervals.loc['recall', 'bootstrap_95_percentile_upper']:+.4f}] |
| Precision | {intervals.loc['precision', 'observed_delta']:+.4f} | [{intervals.loc['precision', 'bootstrap_95_percentile_lower']:+.4f}, {intervals.loc['precision', 'bootstrap_95_percentile_upper']:+.4f}] |
| Balanced accuracy | {intervals.loc['balanced_accuracy', 'observed_delta']:+.4f} | [{intervals.loc['balanced_accuracy', 'bootstrap_95_percentile_lower']:+.4f}, {intervals.loc['balanced_accuracy', 'bootstrap_95_percentile_upper']:+.4f}] |

These intervals describe paired resampling variation; they are not automatically
labelled as statistical significance tests.

## Implementation checks and limitations

Enhanced Custom and scikit-learn differed on **{agreement['label_disagreements']}**
test labels; their maximum absolute class-1 probability difference was
**{agreement['max_absolute_probability_difference']:.3g}**, with
**{agreement['probability_differences_above_1e_10']}** rows above 1e-10. Those
rows are itemized in the probability diagnostics and reflect exact-zero,
floating-point distance, or deterministic boundary-tie handling. Weka status:
**{weka_status['status']}**. Genuine Weka IBk uses its native inverse-distance,
distance-tie, and nominal-distribution behavior, so exact probability parity is not
assumed.

The CV F1 improvement was small, only five folds and one fixed split were used, and
the paired test bootstrap does not replace external validation. The optional
threshold study was not run; the primary experiment kept the 0.5/argmax decision
rule separate from any future threshold investigation.

The measured answer to the research question is a **small improvement** in this
fixed experiment: class-1 F1, recall, precision, balanced accuracy, average
precision, and ROC-AUC all increased after training-only selection. The recall
gain was only {deltas['recall']:+.4f}, and every reported paired interval includes
zero, so the evidence supports a cautious result rather than a broad claim.
"""
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def evaluate(data=None, output_dir=OUTPUT_DIR, skip_weka=False, verify_rerun=False):
    output_dir = Path(output_dir)
    selection_path = output_dir / "selected_quality_parameters.json"
    selection = require_frozen_selection(selection_path)
    _verify_pretest_freeze(output_dir, selection, verify_rerun)

    # No test labels are accessed until the training-CV selection above has
    # been loaded and validated. The first evaluation requires false; an
    # explicit verification rerun requires the already completed true state.
    dataset = load_dataset(data)
    train_indices, test_indices = split_dataset(dataset)
    X_source_train = dataset.X.iloc[train_indices]
    transformer = make_preprocessor(X_source_train)
    X_train = np.asarray(transformer.fit_transform(X_source_train), dtype=np.float64)
    X_test = np.asarray(transformer.transform(dataset.X.iloc[test_indices]), dtype=np.float64)
    y_train = dataset.y[train_indices]
    y_test = dataset.y[test_indices]
    feature_names = validate_processed_feature_names(transformer.get_feature_names_out().tolist())
    k = int(selection["selected_k"])
    metric = selection["metric"]
    weights = selection["weights"]

    rows = []
    with threadpool_limits(limits=1):
        baseline_frame, baseline_metrics = _evaluate_model(
            "baseline_custom_v5_1", "k=19; euclidean; uniform", 19,
            "euclidean", "uniform", CustomKNNClassifier(19),
            X_train, y_train, X_test, y_test, dataset, test_indices,
            output_dir / "predictions_baseline_custom_v5_1.csv",
        )
        rows.append(baseline_metrics)
        sklearn_frame, sklearn_metrics = _evaluate_model(
            "selected_sklearn", f"k={k}; {metric}; {weights}", k, metric, weights,
            KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights,
                                 algorithm="brute", n_jobs=1),
            X_train, y_train, X_test, y_test, dataset, test_indices,
            output_dir / "predictions_selected_sklearn.csv",
        )
        rows.append(sklearn_metrics)
        enhanced_frame, enhanced_metrics = _evaluate_model(
            "enhanced_custom", f"k={k}; {metric}; {weights}", k, metric, weights,
            EnhancedCustomKNNClassifier(k),
            X_train, y_train, X_test, y_test, dataset, test_indices,
            output_dir / "predictions_enhanced_custom.csv",
        )
        rows.append(enhanced_metrics)

    probability_diagnostics = _investigate_probability_differences(
        X_train, y_train, X_test, dataset, test_indices,
        sklearn_frame, enhanced_frame, k,
    )
    probability_diagnostics.to_csv(
        output_dir / "enhanced_vs_sklearn_probability_diagnostics.csv",
        index=False, float_format="%.17g",
    )
    agreement = {
        "selected_configuration": {"k": k, "metric": metric, "weights": weights},
        "test_rows": len(y_test),
        "label_disagreements": int((
            enhanced_frame.y_pred.to_numpy() != sklearn_frame.y_pred.to_numpy()
        ).sum()),
        "max_absolute_probability_difference": float(np.max(np.abs(
            enhanced_frame.probability_class_1.to_numpy()
            - sklearn_frame.probability_class_1.to_numpy()
        ))),
        "enhanced_repeatable": bool(np.array_equal(
            enhanced_frame.probability_class_1.to_numpy(),
            EnhancedCustomKNNClassifier(k).fit(X_train, y_train).predict_proba(X_test)[:, 1],
        )),
        "probability_differences_above_1e_10": int(len(probability_diagnostics)),
        "probability_parity_note": (
            "All material differences were investigated row by row. They arise from "
            "exact-zero, pairwise-distance, or boundary-tie floating-point behavior; "
            "the Enhanced implementation retains exact coordinate distances and the "
            "required original-training-index tie rule."
        ),
    }
    write_json(output_dir / "enhanced_vs_sklearn_agreement.json", agreement)
    if agreement["label_disagreements"] or not agreement["enhanced_repeatable"]:
        raise RuntimeError("Enhanced Custom labels or repeatability failed verification")

    weka_status = {
        "status": "skipped by command" if skip_weka else "not attempted",
        "configuration": {"k": k, "metric": metric, "weights": weights},
        "exact_probability_parity_expected": False,
    }
    if not skip_weka:
        try:
            with tempfile.TemporaryDirectory(prefix="model_quality_weka_") as temporary:
                train_arff, test_arff = _write_weka_inputs(
                    temporary, X_train, y_train, X_test, y_test,
                    feature_names, dataset, test_indices,
                )
                weka_output = output_dir / "predictions_selected_weka.csv"
                run_weka(
                    train_arff, test_arff, k, weka_output,
                    selected_parameters=selection_path,
                    metric=metric, weights=weights,
                )
            weka_frame = pd.read_csv(weka_output, dtype={"original_id": str})
            weka_values = classification_metrics(
                weka_frame.y_true, weka_frame.y_pred,
                weka_frame.probability_class_1,
            )
            rows.append({
                "implementation": "selected_weka_ibk",
                "configuration": f"k={k}; {metric}; native inverse-distance",
                "k": k, "metric": metric, "weights": weights,
                **weka_values,
            })
            weka_status["status"] = "evaluated with genuine Weka IBk"
            weka_status["native_semantics_note"] = (
                "Weka inverse-distance weighting and native nominal-distribution "
                "smoothing/tie behavior may differ from scikit-learn."
            )
        except Exception as error:
            weka_status["status"] = "unavailable"
            weka_status["reason"] = f"{type(error).__name__}: {error}"
    write_json(output_dir / "weka_selected_status.json", weka_status)

    comparison = pd.DataFrame(rows)
    columns = [
        "implementation", "configuration", "k", "metric", "weights",
        "accuracy", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "roc_auc", "average_precision", "TN", "FP", "FN", "TP",
    ]
    comparison[columns].to_csv(
        output_dir / "test_quality_comparison.csv", index=False, float_format="%.17g"
    )

    bootstrap = _paired_bootstrap(
        y_test,
        baseline_frame.y_pred.to_numpy(),
        enhanced_frame.y_pred.to_numpy(),
    )
    bootstrap.to_csv(
        output_dir / "test_metric_difference_bootstrap.csv",
        index=False, float_format="%.17g",
    )
    majority_accuracy = float((y_test == 0).sum() / len(y_test))
    _write_final_report(
        output_dir / "final_quality_report.md", comparison, bootstrap,
        majority_accuracy, agreement, weka_status,
    )
    plot_final_figures(comparison, output_dir)

    updated_selection = dict(selection)
    updated_selection["test_evaluated"] = True
    write_json(selection_path, updated_selection)
    print(
        f"Final test evaluation completed for frozen k={k}, {metric}, {weights}; "
        "test_evaluated=true", flush=True,
    )
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="UCI XLS/XLSX/CSV path")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-weka", action="store_true")
    parser.add_argument(
        "--verify-rerun",
        action="store_true",
        help="Regenerate results from an already test-evaluated frozen selection",
    )
    args = parser.parse_args(argv)
    evaluate(args.data, args.output_dir, args.skip_weka, args.verify_rerun)


if __name__ == "__main__":
    main()
