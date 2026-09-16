"""Explain saved toolkit differences with full-sort neighbours and native voting rules."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.experiment import SCORE_TOLERANCE, array_hash
from src.common.utils import read_json, write_json
from .final import read_predictions


def full_sort_details(training, labels, query, k):
    differences = training - query
    squared = np.einsum("ij,ij->i", differences, differences)
    order = np.lexsort((np.arange(len(training)), squared))
    distances = np.sqrt(squared)
    selected = order[:k]
    zero = distances[selected] == 0
    weights = zero.astype(float) if zero.any() else 1.0 / distances[selected]
    score = float(np.dot(weights, labels[selected]) / weights.sum())
    native = order[distances[order] <= distances[order[k - 1]]]
    native_weights = 1.0 / (distances[native] / np.sqrt(training.shape[1]) + 0.001)
    prior = 1.0 / len(training)
    native_score = float(
        (prior + np.dot(native_weights, labels[native]))
        / (2 * prior + native_weights.sum())
    )
    return distances, selected, score, native, native_score


def diagnose(run_dir):
    """Use this run's prepared matrices and actual scores; do not substitute reconstructed scores."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    data_manifest = read_json(run_dir / "data_manifest.json")
    config = read_json(run_dir / "resolved_configs.json")["final"]
    frames = {
        name: read_predictions(run_dir / "predictions" / f"{name}.csv")
        for name in ("final_python", "final_sklearn", "final_weka")
    }
    with np.load(run_dir / "data/final/train.npz", allow_pickle=False) as stored:
        training, labels, train_ids = stored["X"], stored["y"], stored["row_ids"]
    with np.load(run_dir / "data/final/test.npz", allow_pickle=False) as stored:
        queries = stored["X"]
    for array, key in (
        (training, "train_numeric_hash"),
        (queries, "test_numeric_hash"),
    ):
        if array_hash(array) != data_manifest["groups"]["final"][key]:
            raise ValueError("Diagnostic input differs from measured prepared matrix")
    manual = frames["final_python"]
    for frame in frames.values():
        if (
            not (frame.run_id == manifest["run_id"]).all()
            or not (frame.config_hash == config["config_hash"]).all()
        ):
            raise ValueError("Stale diagnostic prediction input")
        if not np.array_equal(frame.row_id, manual.row_id):
            raise ValueError("Diagnostic row alignment differs")
    sklearn_delta = np.abs(
        manual.score_class_1 - frames["final_sklearn"].score_class_1
    ).to_numpy()
    weka_delta = np.abs(
        manual.score_class_1 - frames["final_weka"].score_class_1
    ).to_numpy()
    sklearn_rows = np.flatnonzero(sklearn_delta > SCORE_TOLERANCE)
    weka_rows = np.union1d(
        np.flatnonzero(manual.y_pred != frames["final_weka"].y_pred),
        np.argsort(weka_delta)[-5:],
    )
    selected_rows = np.union1d(sklearn_rows, weka_rows)
    k = config["model"]["k"]
    toolkit = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="euclidean",
        algorithm="brute",
        n_jobs=1,
    ).fit(training, labels)
    neighbour_rows, cases = [], []
    for position in selected_rows:
        distances, chosen, score, native, reconstructed = full_sort_details(
            training, labels, queries[position], k
        )
        sklearn_distances, sklearn_indices = toolkit.kneighbors(
            queries[position : position + 1]
        )
        sklearn_indices = sklearn_indices[0]
        if abs(score - manual.score_class_1.iloc[position]) > SCORE_TOLERANCE:
            raise ValueError(
                "Full-sort diagnostic oracle disagrees with measured manual score"
            )
        entry = {
            "test_position": int(position),
            "row_id": int(manual.row_id.iloc[position]),
            "original_id": str(manual.original_id.iloc[position]),
            "manual_score": float(score),
            "y_true": int(manual.y_true.iloc[position]),
            "threshold": config["model"]["threshold"],
            "manual_label": int(manual.y_pred.iloc[position]),
            "sklearn_label": int(frames["final_sklearn"].y_pred.iloc[position]),
            "weka_label": int(frames["final_weka"].y_pred.iloc[position]),
            "manual_threshold_margin": float(score - config["model"]["threshold"]),
            "weka_threshold_margin": float(
                frames["final_weka"].score_class_1.iloc[position]
                - config["model"]["threshold"]
            ),
            "sklearn_score": float(
                frames["final_sklearn"].score_class_1.iloc[position]
            ),
            "weka_score": float(frames["final_weka"].score_class_1.iloc[position]),
            "weka_native_formula_reconstruction": reconstructed,
            "weka_reconstruction_error": abs(
                reconstructed - frames["final_weka"].score_class_1.iloc[position]
            ),
            "zero_distance_training_rows": int(np.count_nonzero(distances == 0)),
            "manual_selected_count": len(chosen),
            "reconstructed_weka_boundary_count": len(native),
            "manual_only_training_positions": np.setdiff1d(
                chosen, sklearn_indices
            ).tolist(),
            "sklearn_only_training_positions": np.setdiff1d(
                sklearn_indices, chosen
            ).tolist(),
        }
        entry["sklearn_cause"] = (
            "boundary neighbour selection"
            if entry["manual_only_training_positions"]
            else "distance/vote arithmetic on the same neighbours"
        )
        cases.append(entry)
        for rank, index in enumerate(native):
            neighbour_rows.append(
                {
                    "test_position": int(position),
                    "row_id": entry["row_id"],
                    "rank": rank,
                    "training_position": int(index),
                    "training_row_id": int(train_ids[index]),
                    "label": int(labels[index]),
                    "euclidean_distance": float(distances[index]),
                    "selected_manual": bool(index in chosen),
                    "selected_sklearn": bool(index in sklearn_indices),
                    "zero_distance": bool(distances[index] == 0),
                }
            )
    output = run_dir / "diagnostics"
    pd.DataFrame(neighbour_rows).to_csv(
        output / "disputed_neighbours.csv", index=False, float_format="%.17g"
    )
    report = {
        "run_id": manifest["run_id"],
        "config_hash": config["config_hash"],
        "tolerance": SCORE_TOLERANCE,
        "sklearn_rows_over_tolerance": len(sklearn_rows),
        "inspected_weka_rows": weka_rows.tolist(),
        "scope": "Full-sort inspection of saved current-run scores; Weka formula reconstruction uses independently computed distances and never replaces genuine recorded IBk output. Reconstruction errors, including arithmetic or boundary differences, are reported explicitly.",
        "weka_formula": "(1/n_train + sum(y / (distance/sqrt(d)+0.001))) / (2/n_train + sum(1/(distance/sqrt(d)+0.001))); include kth-distance ties",
        "cases": cases,
    }
    write_json(output / "numerical_investigation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with threadpool_limits(1):
        report = diagnose(args.run_dir)
    print(f"Investigated {len(report['cases'])} current-run cases")


if __name__ == "__main__":
    main()
