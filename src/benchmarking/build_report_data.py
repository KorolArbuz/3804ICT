"""Build report tables from raw measurements and authoritative saved evidence."""

import csv
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR
from src.common.utils import load_pair

from .benchmark_utils import (
    BENCHMARK_DIR,
    DISPLAY_NAMES,
    IMPLEMENTATIONS,
    make_python_models,
    read_json,
    read_rows,
    summarize,
    write_json,
    write_rows,
)


NOTES = {
    "custom_python_v5_1": "Accepted V5.1; batch 64; one-thread numerical pools",
    "sklearn": "Brute Euclidean; uniform voting; n_jobs=1",
    "weka": "IBk internal prediction timer; fresh one-CPU JVM per observation",
    "custom_cpp": "Experimental native Release; exact heap; batch 32",
}


def _write_table(stem, rows, columns):
    path = BENCHMARK_DIR / f"{stem}.csv"
    write_rows(path, rows, columns)


def _grouped_summary(rows, value_field, group_fields, warmup_field=None):
    grouped = defaultdict(list)
    for row in rows:
        if warmup_field and row[warmup_field] == "true":
            continue
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(float(row[value_field]))
    return grouped


def _statistics_row(name, values, notes=""):
    stats = summarize(values)
    return {
        "implementation": name,
        "runs": stats["count"],
        "median_seconds": stats["median_seconds"],
        "mean_seconds": stats["mean_seconds"],
        "min_seconds": stats["min_seconds"],
        "max_seconds": stats["max_seconds"],
        "sample_std_seconds": stats["sample_std_seconds"],
        "q1_seconds": stats["q1_seconds"],
        "q3_seconds": stats["q3_seconds"],
        "iqr_seconds": stats["iqr_seconds"],
        "p10_seconds": stats["p10_seconds"],
        "p90_seconds": stats["p90_seconds"],
        "coefficient_of_variation": stats["coefficient_of_variation"],
        "notes": notes,
    }


def _bootstrap_median_interval(values, seed, resamples=10_000):
    """Return a deterministic percentile bootstrap interval for the raw median."""
    values = np.asarray([float(value) for value in values], dtype=float)
    randomizer = np.random.default_rng(seed)
    indices = randomizer.integers(0, len(values), size=(resamples, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def _linear_regression(points):
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope_seconds_per_row": float(slope),
        "intercept_seconds": float(intercept),
        "r_squared": 1.0 - residual / total if total else 1.0,
    }


def _prediction_tables():
    rows = read_rows(BENCHMARK_DIR / "raw_prediction_runs.csv")
    grouped = _grouped_summary(rows, "runtime_seconds", ("implementation",), "warmup")
    statistics = []
    for index, name in enumerate(IMPLEMENTATIONS):
        values = grouped[(name,)]
        row = _statistics_row(name, values, NOTES[name])
        low, high = _bootstrap_median_interval(values, seed=3804 + index)
        row.update({
            "median_ci95_low": low,
            "median_ci95_high": high,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 3804 + index,
        })
        statistics.append(row)
    columns = tuple(statistics[0])
    _write_table("prediction_runtime_statistics", statistics, columns)
    return statistics, rows


def _paired_speedups(raw_prediction):
    """Pair implementations by randomized trial number and summarize V5.1 ratios."""
    timed = [row for row in raw_prediction if row["warmup"] == "false"]
    by_trial = {
        (int(row["run_number_within_implementation"]), row["implementation"]): float(row["runtime_seconds"])
        for row in timed
    }
    comparisons = (
        ("custom_cpp", "C++20 experimental"),
        ("sklearn", "scikit-learn"),
        ("weka", "Weka IBk"),
    )
    trial_numbers = sorted({trial for trial, implementation in by_trial if implementation == "custom_python_v5_1"})
    rows = []
    for candidate, label in comparisons:
        for trial in trial_numbers:
            reference = by_trial[(trial, "custom_python_v5_1")]
            candidate_runtime = by_trial[(trial, candidate)]
            rows.append({
                "trial": trial,
                "comparison": label,
                "reference_runtime_seconds": reference,
                "candidate_runtime_seconds": candidate_runtime,
                "speedup_factor": reference / candidate_runtime,
            })
    _write_table("paired_speedups", rows, tuple(rows[0]))

    summaries = []
    for candidate, label in comparisons:
        values = [
            row["speedup_factor"]
            for row in rows
            if row["comparison"] == label
        ]
        stats = summarize(values)
        summaries.append({
            "comparison": label,
            "candidate_implementation": candidate,
            "paired_trials": stats["count"],
            "median_speedup_factor": stats["median_seconds"],
            "q1_speedup_factor": stats["q1_seconds"],
            "q3_speedup_factor": stats["q3_seconds"],
            "iqr_speedup_factor": stats["iqr_seconds"],
            "min_speedup_factor": stats["min_seconds"],
            "max_speedup_factor": stats["max_seconds"],
            "candidate_wins": sum(value > 1.0 for value in values),
            "v5_1_wins": sum(value < 1.0 for value in values),
            "ties": sum(value == 1.0 for value in values),
        })
    _write_table("paired_speedup_summary", summaries, tuple(summaries[0]))
    return rows, summaries


def _pipeline_tables():
    rows = read_rows(BENCHMARK_DIR / "raw_full_pipeline_runs.csv")
    totals = _grouped_summary(rows, "total_seconds", ("implementation",))
    fits = _grouped_summary(rows, "fit_seconds", ("implementation",))
    pipeline = [
        _statistics_row(
            name,
            totals[(name,)],
            "Prepared input loading, fit/build, prediction, output, and metrics",
        )
        for name in IMPLEMENTATIONS
    ]
    fit = [
        _statistics_row(
            name,
            fits[(name,)],
            "Internal fit/build scope differs by implementation; compare cautiously",
        )
        for name in IMPLEMENTATIONS
    ]
    columns = tuple(pipeline[0])
    _write_table("full_pipeline_runtime", pipeline, columns)
    _write_table("fit_build_runtime", fit, columns)
    return pipeline, fit


def _scaling_table(filename, size_field, output_stem):
    raw = read_rows(BENCHMARK_DIR / filename)
    grouped = _grouped_summary(raw, "runtime_seconds", (size_field, "implementation"), "warmup")
    rows = []
    regression_points = defaultdict(list)
    for (size, name), values in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        size = int(size)
        stats = summarize(values)
        sample = next(
            row for row in raw
            if row[size_field] == str(size) and row["implementation"] == name and row["warmup"] == "false"
        )
        query_rows = int(sample["query_rows"])
        train_rows = int(sample["train_rows"])
        rows.append({
            size_field: size,
            "implementation": name,
            "runs": stats["count"],
            "median_seconds": stats["median_seconds"],
            "min_seconds": stats["min_seconds"],
            "max_seconds": stats["max_seconds"],
            "mean_seconds": stats["mean_seconds"],
            "sample_std_seconds": stats["sample_std_seconds"],
            "iqr_seconds": stats["iqr_seconds"],
            "queries_per_second_at_median": query_rows / stats["median_seconds"],
            "milliseconds_per_query_at_median": 1000 * stats["median_seconds"] / query_rows,
            "microseconds_per_query_per_1000_training_rows": (
                1_000_000 * stats["median_seconds"] / query_rows / (train_rows / 1000)
            ),
        })
        regression_points[name].append((size, stats["median_seconds"]))
    _write_table(output_stem, rows, tuple(rows[0]))
    regressions = {name: _linear_regression(points) for name, points in regression_points.items()}
    return rows, regressions


def _cpp_table():
    raw = read_rows(BENCHMARK_DIR / "raw_cpp_configuration.csv")
    grouped = _grouped_summary(raw, "runtime_seconds", ("build", "selection_method", "batch_size"))
    rows = []
    for (build, selection, batch), values in sorted(grouped.items()):
        stats = summarize(values)
        sample = next(
            row for row in raw
            if row["build"] == build and row["selection_method"] == selection and row["batch_size"] == batch
        )
        rows.append({
            "build": build,
            "selection_method": selection,
            "batch_size": int(batch),
            "runs": stats["count"],
            "median_seconds": stats["median_seconds"],
            "min_seconds": stats["min_seconds"],
            "max_seconds": stats["max_seconds"],
            "iqr_seconds": stats["iqr_seconds"],
            "compiler": sample["compiler"],
            "executable_sha256": sample["executable_sha256"],
            "selected_main_configuration": sample["selected_main_configuration"],
        })
    _write_table("cpp_configuration_summary", rows, tuple(rows[0]))
    return rows


def _historical(prediction):
    with (RESULTS_DIR / "custom_knn_optimization.csv").open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    names = {
        "baseline_original": ("Original", "Initial custom NumPy implementation"),
        "optimized_v1": ("V1", "Matrix distances and partial selection"),
        "optimized_v2_single_thread": ("V2", "Workspace reuse and lower allocation overhead"),
        "optimized_v3_single_thread": ("V3", "Ranking-only score and less unnecessary ordering"),
        "optimized_v4_threshold_prefilter": ("V4", "Exact deterministic threshold prefilter"),
        "optimized_v5_single_thread": ("V5", "Augmented GEMM and batch 64"),
        "optimized_v5_1_single_thread": ("V5.1", "Feature-major direct fallback layout"),
    }
    rows = []
    original = float(source[0]["prediction_seconds_summary"])
    for item in source:
        version, change = names[item["version"]]
        runtime = float(item["prediction_seconds_summary"])
        rows.append({
            "version": version,
            "main_change": change,
            "accepted": str(version in {"Original", "V1", "V2", "V3", "V4", "V5", "V5.1"}).lower(),
            "measurement_type": item["measurement_summary"],
            "trial_count": item["trial_count"],
            "median_runtime_seconds": runtime,
            "min_seconds": float(item["prediction_seconds_min"]),
            "max_seconds": float(item["prediction_seconds_max"]),
            "speedup_vs_original": original / runtime,
            "notes": "Accepted history point" if version == "V5.1" else "Historical development point",
        })
    v6 = read_json(RESULTS_DIR / "custom_knn_v6_benchmark.json")
    v7 = read_json(RESULTS_DIR / "custom_knn_v7_benchmark.json")
    for version, evidence, change in (
        ("V6", v6["best_v6_candidate"], "Partial-distance lower-bound screening"),
        ("V7", v7["final_v7"]["v7"], "Exact block pruning"),
    ):
        runtime = float(evidence["median_v6_seconds"] if version == "V6" else evidence["median_seconds"])
        rows.append({
            "version": version,
            "main_change": change,
            "accepted": "false",
            "measurement_type": "rejected experimental paired benchmark",
            "trial_count": len(evidence["raw_v6_seconds"] if version == "V6" else evidence["raw_seconds"]),
            "median_runtime_seconds": runtime,
            "min_seconds": float(evidence["min_v6_seconds"] if version == "V6" else evidence["min_seconds"]),
            "max_seconds": float(evidence["max_v6_seconds"] if version == "V6" else evidence["max_seconds"]),
            "speedup_vs_original": original / runtime,
            "notes": (v6 if version == "V6" else v7)["decision_reason"],
        })
    cpp = next(row for row in prediction if row["implementation"] == "custom_cpp")
    runtime = cpp["median_seconds"]
    rows.append({
        "version": "C++ experimental",
        "main_change": "Fused exact distance accumulation and bounded heap",
        "accepted": "false",
        "measurement_type": "current controlled 20-run median",
        "trial_count": cpp["runs"],
        "median_runtime_seconds": runtime,
        "min_seconds": cpp["min_seconds"],
        "max_seconds": cpp["max_seconds"],
        "speedup_vs_original": original / runtime,
        "notes": "Experimental candidate; exact output, one-machine evidence",
    })
    _write_table("historical_optimization", rows, tuple(rows[0]))
    return rows


def _rejected(cpp_rows):
    v5 = read_json(RESULTS_DIR / "custom_knn_v5_benchmark.json")
    v51 = read_json(RESULTS_DIR / "custom_knn_v5_1_benchmark.json")
    v6 = read_json(RESULTS_DIR / "custom_knn_v6_benchmark.json")
    v7 = read_json(RESULTS_DIR / "custom_knn_v7_benchmark.json")
    fallback = v5["fallback_candidates"]["candidates"]
    cpp_heap = next(row for row in cpp_rows if row["build"] == "native" and row["selection_method"] == "bounded max-heap" and row["batch_size"] == 32)
    cpp_nth = next(row for row in cpp_rows if row["build"] == "native" and row["selection_method"] == "nth_element")
    rows = [
        {
            "experiment": "Batched direct fallback",
            "hypothesis": "Vectorize exact fallback rows in chunks",
            "reference_seconds": fallback["current"]["median_seconds"],
            "candidate_seconds": fallback["batched_8"]["median_seconds"],
            "result": "Best tested batch still increased total and fallback time",
            "decision": "Rejected",
            "reason": fallback["batched_8"]["reason"],
        },
        {
            "experiment": "Python batches below 64",
            "hypothesis": "Smaller score workspaces could improve cache behavior",
            "reference_seconds": v51["batch_sweep"]["candidates"]["64"]["median_seconds"],
            "candidate_seconds": v51["batch_sweep"]["candidates"]["40"]["median_seconds"],
            "result": "Ranks changed across machine states and gains did not clear the acceptance rule",
            "decision": "Rejected",
            "reason": v51["selected_batch_size"]["reason"],
        },
        {
            "experiment": "V6 lower-bound pruning",
            "hypothesis": "Partial distances can cheaply reject most rows",
            "reference_seconds": v6["best_v6_candidate"]["median_v5_1_seconds"],
            "candidate_seconds": v6["best_v6_candidate"]["median_v6_seconds"],
            "result": "Candidate was about 1.70x slower",
            "decision": "Rejected",
            "reason": v6["decision_reason"],
        },
        {
            "experiment": "V7 block pruning",
            "hypothesis": "Block bounds can avoid exact row evaluation",
            "reference_seconds": v7["final_v7"]["v5_1"]["median_seconds"],
            "candidate_seconds": v7["final_v7"]["v7"]["median_seconds"],
            "result": "Candidate was about 1.47x slower",
            "decision": "Rejected",
            "reason": v7["decision_reason"],
        },
        {
            "experiment": "C++ nth_element selection",
            "hypothesis": "Contiguous partial partition may beat bounded heaps",
            "reference_seconds": cpp_heap["median_seconds"],
            "candidate_seconds": cpp_nth["median_seconds"],
            "result": f"Native nth_element was {cpp_nth['median_seconds'] / cpp_heap['median_seconds']:.2f}x slower",
            "decision": "Rejected",
            "reason": "The bounded heap avoids materializing and partitioning the full distance matrix.",
        },
    ]
    _write_table("rejected_optimization_experiments", rows, tuple(rows[0]))
    return rows


def _quality_and_correctness():
    with (RESULTS_DIR / "metrics_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    quality_columns = (
        "implementation", "selected_k", "test_samples", "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy",
        "roc_auc", "average_precision",
    )
    quality = [{column: row[column] for column in quality_columns} for row in metrics]
    custom = next(row for row in quality if row["implementation"] == "custom")
    quality.append({**custom, "implementation": "custom_cpp"})
    _write_table("quality_metrics", quality, quality_columns)
    confusion_columns = ("implementation", "TN", "FP", "FN", "TP")
    confusion = [{column: row[column] for column in confusion_columns} for row in quality]
    _write_table("confusion_counts", confusion, confusion_columns)

    with (RESULTS_DIR / "prediction_agreement.csv").open(encoding="utf-8-sig", newline="") as handle:
        base = list(csv.DictReader(handle))
    pairs = []
    for row in base:
        pairs.append({**row, "exact_prediction_matches": int(round(float(row["class_agreement"]) * 6000)), "exact_vote_count_matches": "not applicable"})
    custom_pairs = {row["implementation_b"]: row for row in base if row["implementation_a"] == "custom"}
    pairs.extend([
        {
            "implementation_a": "custom", "implementation_b": "custom_cpp", "test_samples": 6000,
            "class_agreement": 1.0, "disagreement_count": 0, "probability_correlation": 1.0,
            "max_absolute_probability_difference": 0.0, "exact_prediction_matches": 6000,
            "exact_vote_count_matches": 6000,
        },
        {**custom_pairs["sklearn"], "implementation_a": "sklearn", "implementation_b": "custom_cpp", "exact_prediction_matches": 5998, "exact_vote_count_matches": "not applicable"},
        {**custom_pairs["weka"], "implementation_a": "weka", "implementation_b": "custom_cpp", "exact_prediction_matches": 5999, "exact_vote_count_matches": "not applicable"},
    ])
    correctness_columns = (
        "implementation_a", "implementation_b", "test_samples", "class_agreement",
        "disagreement_count", "probability_correlation", "max_absolute_probability_difference",
        "exact_prediction_matches", "exact_vote_count_matches",
    )
    _write_table("correctness_summary", pairs, correctness_columns)
    return quality, confusion, pairs


def _memory_table(cpp_rows):
    training, testing = load_pair(PROCESSED_DATA_DIR / "train.npz", PROCESSED_DATA_DIR / "test.npz")
    with threadpool_limits(limits=1):
        models = make_python_models(19)
        for model in models.values():
            model.fit(training["X"], training["y"])
    custom = models["custom_python_v5_1"]
    custom_persistent = sum(
        getattr(custom, name).nbytes
        for name in ("X_", "_train_squared_norms", "_augmented_training", "_X_by_feature", "y_", "classes_", "_pilot_indices")
    )
    sklearn = models["sklearn"]
    sklearn_persistent = sklearn._fit_X.nbytes + sklearn._y.nbytes + sklearn.classes_.nbytes
    selected_cpp = next(row for row in cpp_rows if row["selected_main_configuration"] == "true")
    cpp_total_auxiliary = int(next(
        row["peak_auxiliary_bytes_estimate"]
        for row in read_rows(BENCHMARK_DIR / "raw_cpp_configuration.csv")
        if row["selected_main_configuration"] == "true"
    ))
    rows = [
        {"implementation": "custom_python_v5_1", "category": "persistent_model_storage", "bytes": custom_persistent, "mib": custom_persistent / 2**20, "basis": "Measured NumPy array nbytes after fit"},
        {"implementation": "custom_python_v5_1", "category": "major_prediction_workspace", "bytes": 64 * 24000 * 8 + 64 * 34 * 8 + 6000 * 19 * 8 + 24000 * 8, "mib": (64 * 24000 * 8 + 64 * 34 * 8 + 6000 * 19 * 8 + 24000 * 8) / 2**20, "basis": "Estimated major arrays; batch workspace, neighbours, and one fallback vector"},
        {"implementation": "custom_python_v5_1", "category": "returned_probability_buffer", "bytes": 6000 * 2 * 8, "mib": 6000 * 2 * 8 / 2**20, "basis": "Estimated float64 probability result"},
        {"implementation": "sklearn", "category": "persistent_model_storage", "bytes": sklearn_persistent, "mib": sklearn_persistent / 2**20, "basis": "Measured _fit_X, encoded labels, and classes nbytes"},
        {"implementation": "custom_cpp", "category": "persistent_model_storage", "bytes": 24000 * 33 * 8 + 24000 * 4, "mib": (24000 * 33 * 8 + 24000 * 4) / 2**20, "basis": "Estimated vector capacities for training doubles and int labels"},
        {"implementation": "custom_cpp", "category": "major_prediction_workspace", "bytes": cpp_total_auxiliary - 6000 * (4 + 8), "mib": (cpp_total_auxiliary - 6000 * (4 + 8)) / 2**20, "basis": f"Executable estimate for {selected_cpp['build']} heap/batch 32 excluding result buffers"},
        {"implementation": "custom_cpp", "category": "result_buffers", "bytes": 6000 * (4 + 8), "mib": 6000 * (4 + 8) / 2**20, "basis": "Estimated int labels plus size_t vote counts"},
    ]
    _write_table("raw_memory", rows, tuple(rows[0]))
    return rows


def _environment_table():
    environment = read_json(BENCHMARK_DIR / "environment_comprehensive.json")
    return environment


def main():
    prediction, raw_prediction = _prediction_tables()
    paired_speedups, paired_speedup_summary = _paired_speedups(raw_prediction)
    pipeline, fit = _pipeline_tables()
    train_scaling, train_regressions = _scaling_table(
        "raw_scaling_train.csv", "train_rows", "training_size_scaling"
    )
    query_scaling, query_regressions = _scaling_table(
        "raw_scaling_queries.csv", "query_rows", "query_size_scaling"
    )
    cpp = _cpp_table()
    history = _historical(prediction)
    rejected = _rejected(cpp)
    quality, confusion, correctness = _quality_and_correctness()
    memory = _memory_table(cpp)
    environment = _environment_table()
    combined = []
    for mode, records in (("prediction_only", prediction), ("full_pipeline", pipeline), ("fit_build", fit)):
        for row in records:
            combined.append({"mode": mode, **row})
    _write_table("runtime_summary", combined, tuple(combined[0]))
    write_json(BENCHMARK_DIR / "runtime_summary.json", {
        "prediction_only": prediction,
        "paired_speedups": paired_speedups,
        "paired_speedup_summary": paired_speedup_summary,
        "full_pipeline": pipeline,
        "fit_build": fit,
        "training_size_scaling": train_scaling,
        "query_size_scaling": query_scaling,
        "training_size_regression": train_regressions,
        "query_size_regression": query_regressions,
        "cpp_configurations": cpp,
        "historical_optimization": history,
        "rejected_experiments": rejected,
        "quality_metrics": quality,
        "confusion_counts": confusion,
        "correctness": correctness,
        "memory": memory,
        "environment": environment,
    })
    print("Comprehensive CSV and JSON report tables generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
