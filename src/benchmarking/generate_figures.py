"""Generate comprehensive report figures from saved CSV/JSON artifacts only."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.config import RESULTS_DIR

from .benchmark_utils import BENCHMARK_DIR, COLORS, DISPLAY_NAMES, FIGURES_DIR, IMPLEMENTATIONS, read_json, read_rows


SHORT_NAMES = {
    "custom_python_v5_1": "Python V5.1",
    "sklearn": "scikit-learn",
    "weka": "Weka IBk",
    "custom_cpp": "C++20\nexperimental",
}


def _save(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES_DIR / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def _bar(values, title, ylabel, stem, labels=None, colors=None, value_format="{:.3f}"):
    labels = labels or [SHORT_NAMES[name] for name in IMPLEMENTATIONS]
    colors = colors or [COLORS[name] for name in IMPLEMENTATIONS]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[value_format.format(value) for value in values], padding=4, fontsize=9)
    _save(fig, stem)


def _prediction_figures(summary):
    medians = {row["implementation"]: row["median_seconds"] for row in summary["prediction_only"]}
    values = [medians[name] for name in IMPLEMENTATIONS]
    _bar(
        values,
        "Prediction runtime for one complete test-set pass",
        "Median prediction time (seconds)",
        "prediction_runtime_controlled",
    )
    raw = read_rows(BENCHMARK_DIR / "raw_prediction_runs.csv")
    grouped = {
        name: [float(row["runtime_seconds"]) for row in raw if row["implementation"] == name and row["warmup"] == "false"]
        for name in IMPLEMENTATIONS
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    plot = ax.boxplot(
        [grouped[name] for name in IMPLEMENTATIONS],
        labels=[SHORT_NAMES[name] for name in IMPLEMENTATIONS],
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5},
    )
    for patch, name in zip(plot["boxes"], IMPLEMENTATIONS):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.78)
    ax.set_yscale("log")
    ax.set_title("Distribution of prediction runtimes", fontsize=13, pad=12)
    ax.set_ylabel("Prediction time (seconds, logarithmic scale)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "prediction_runtime_distribution_comprehensive")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for name in IMPLEMENTATIONS:
        points = [row for row in raw if row["implementation"] == name and row["warmup"] == "false"]
        ax.plot(
            [int(row["global_run_order"]) for row in points],
            [float(row["runtime_seconds"]) for row in points],
            marker="o", markersize=3.8, linewidth=1.2,
            label=SHORT_NAMES[name].replace("\n", " "), color=COLORS[name],
        )
    ax.set_title("Prediction runtime by benchmark run order", fontsize=13, pad=12)
    ax.set_xlabel("Global interleaved run order")
    ax.set_ylabel("Prediction time (seconds)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    _save(fig, "prediction_runtime_run_order_comprehensive")

    throughput = [6000 / medians[name] for name in IMPLEMENTATIONS]
    _bar(
        throughput,
        "Prediction throughput",
        "Queries per second (higher is better)",
        "prediction_throughput",
        value_format="{:.0f}",
    )
    milliseconds = [1000 * medians[name] / 6000 for name in IMPLEMENTATIONS]
    _bar(
        milliseconds,
        "Average prediction time per query",
        "Milliseconds per query",
        "milliseconds_per_query",
        value_format="{:.3f}",
    )


def _pipeline_figures(summary):
    pipeline = {row["implementation"]: row["median_seconds"] for row in summary["full_pipeline"]}
    fit = {row["implementation"]: row["median_seconds"] for row in summary["fit_build"]}
    _bar(
        [pipeline[name] for name in IMPLEMENTATIONS],
        "End-to-end experiment runtime",
        "Median prepared-input pipeline time (seconds)",
        "full_pipeline_runtime_comprehensive",
    )
    _bar(
        [fit[name] for name in IMPLEMENTATIONS],
        "Model fit/build time",
        "Median fit/build time (seconds)",
        "fit_build_runtime",
    )


def _history_figures(summary):
    accepted = [
        row for row in summary["historical_optimization"]
        if row["accepted"] == "true" or row["version"] == "C++ experimental"
    ]
    labels = [row["version"] for row in accepted]
    runtimes = [float(row["median_runtime_seconds"]) for row in accepted]
    colors = ["#4c78a8"] * (len(labels) - 1) + [COLORS["custom_cpp"]]
    for scale, stem in (("linear", "custom_optimization_history_linear"), ("log", "custom_optimization_history_log")):
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.plot(labels, runtimes, color="#4c78a8", marker="o", linewidth=1.8)
        ax.scatter(labels[-1], runtimes[-1], color=colors[-1], zorder=3, label="Experimental C++")
        for x, runtime, row in zip(labels, runtimes, accepted):
            ax.annotate(f"{row['speedup_vs_original']:.1f}x", (x, runtime), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title("Custom KNN optimization history", fontsize=13, pad=12)
        ax.set_ylabel("Prediction time (seconds)" + (" — logarithmic scale" if scale == "log" else ""))
        ax.set_xlabel("Implementation version")
        if scale == "linear":
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale("log")
            ax.set_ylim(min(runtimes) * 0.65, max(runtimes) * 1.7)
        ax.grid(axis="y", alpha=0.25)
        if scale == "log":
            ax.legend(fontsize=9)
        _save(fig, stem)
    _bar(
        [row["speedup_vs_original"] for row in accepted],
        "Speedup relative to the original custom implementation",
        "Recorded speedup factor (higher is faster)",
        "speedup_vs_original",
        labels=labels,
        colors=colors,
        value_format="{:.1f}x",
    )
    prediction = {row["implementation"]: row["median_seconds"] for row in summary["prediction_only"]}
    baseline = prediction["custom_python_v5_1"]
    _bar(
        [baseline / prediction[name] for name in IMPLEMENTATIONS],
        "Relative prediction performance vs Python V5.1",
        "Speedup = V5.1 runtime / implementation runtime (>1 is faster)",
        "speedup_vs_v5_1_comprehensive",
        value_format="{:.2f}x",
    )


def _phase_figure():
    profile = read_json(Path("data/processed/v5_local/final_profile.json"))["median_phase_seconds"]
    phases = [
        ("Augmented GEMM / scoring", profile["augmented_query_and_gemm_including_query_norms"]),
        ("Pilot threshold", profile["pilot_threshold"]),
        ("Threshold scan", profile["full_scan"]),
        ("Reduced partition", profile["reduced_partition"]),
        ("Boundary detection", profile["boundary_detection"]),
        ("Direct fallback", profile["direct_fallback"]),
    ]
    included = sum(value for _, value in phases)
    phases.append(("Other measured phases", max(0.0, sum(profile.values()) - included)))
    labels, values = zip(*reversed(phases))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.barh(labels, values, color="#4c78a8", edgecolor="#333333", linewidth=0.5)
    ax.set_title("V5 prediction phase breakdown", fontsize=13, pad=12)
    ax.set_xlabel("Median phase time (seconds)")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    _save(fig, "v5_phase_breakdown")


def _cpp_figures(summary):
    rows = summary["cpp_configurations"]
    labels = [f"{row['build'].title()}\n{'Heap' if row['selection_method'].startswith('bounded') else 'nth_element'} / {row['batch_size']}" for row in rows]
    colors = ["#e15759" if row["selected_main_configuration"] == "true" else "#9c9c9c" for row in rows]
    values = [row["median_seconds"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    bars = ax.barh(list(reversed(labels)), list(reversed(values)), color=list(reversed(colors)), edgecolor="#444444", linewidth=0.6)
    ax.set_title("Pure C++20 configuration screening", fontsize=13, pad=12)
    ax.set_xlabel("Median prediction time (seconds)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in reversed(values)], padding=3, fontsize=8)
    _save(fig, "cpp_configuration_screen")
    heap32 = [row for row in rows if row["selection_method"] == "bounded max-heap" and row["batch_size"] == 32]
    _bar(
        [row["median_seconds"] for row in heap32],
        "C++ Release build comparison",
        "Median prediction time (seconds)",
        "cpp_build_comparison",
        labels=[row["build"].title() for row in heap32],
        colors=["#9c9c9c", "#e15759"],
    )


def _scaling_figure(rows, size_field, title, stem):
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for name in IMPLEMENTATIONS:
        points = sorted((row for row in rows if row["implementation"] == name), key=lambda row: row[size_field])
        ax.plot(
            [row[size_field] for row in points], [row["median_seconds"] for row in points],
            marker="o", linewidth=1.7, label=SHORT_NAMES[name].replace("\n", " "), color=COLORS[name],
        )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(size_field.replace("_", " ").title())
    ax.set_ylabel("Median prediction time (seconds)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    _save(fig, stem)


def _quality_figures(summary):
    runtimes = {row["implementation"]: row["median_seconds"] for row in summary["prediction_only"]}
    quality = {row["implementation"]: row for row in summary["quality_metrics"]}
    quality["custom_python_v5_1"] = quality["custom"]
    for metric, title, stem in (
        ("f1", "Runtime vs classification F1", "runtime_vs_f1"),
        ("accuracy", "Runtime vs classification accuracy", "runtime_vs_accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for name in IMPLEMENTATIONS:
            metric_name = "custom_cpp" if name == "custom_cpp" else name
            point = quality[metric_name]
            ax.scatter(runtimes[name], float(point[metric]), s=70, color=COLORS[name], edgecolor="#333333")
            ax.annotate(SHORT_NAMES[name].replace("\n", " "), (runtimes[name], float(point[metric])), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("Median prediction time (seconds)")
        ax.set_ylabel(metric.upper() if metric == "f1" else "Accuracy")
        ax.grid(alpha=0.25)
        _save(fig, stem)

    custom = quality["custom"]
    matrix = np.array([[int(custom["TN"]), int(custom["FP"])], [int(custom["FN"]), int(custom["TP"])]])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value:,}", ha="center", va="center", color="white" if value > matrix.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title("Accepted Python V5.1 confusion matrix", fontsize=13, pad=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "confusion_matrix_v5_1")

    names = ["custom", "sklearn", "weka", "custom_cpp"]
    labels = ["Python V5.1", "scikit-learn", "Weka", "C++20"]
    matrix = np.eye(4)
    for row in summary["correctness"]:
        a, b = row["implementation_a"], row["implementation_b"]
        i, j = names.index(a), names.index(b)
        matrix[i, j] = matrix[j, i] = float(row["class_agreement"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix * 100, cmap="Blues", vmin=99.9, vmax=100.0)
    for (row, column), value in np.ndenumerate(matrix * 100):
        ax.text(column, row, f"{value:.3f}%", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_yticks(range(4), labels)
    ax.set_title("Pairwise prediction agreement", fontsize=13, pad=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Agreement (%)")
    _save(fig, "implementation_agreement_heatmap")


def _cv_figures():
    with (RESULTS_DIR / "cv_results_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    k = np.asarray([int(row["k"]) for row in rows])
    for metric, title, stem in (
        ("f1", "Cross-validation F1 by number of neighbours", "cv_f1_by_k_comprehensive"),
        ("balanced_accuracy", "Cross-validation balanced accuracy by number of neighbours", "cv_balanced_accuracy_by_k"),
    ):
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        std = np.asarray([float(row[f"{metric}_std"]) for row in rows])
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        ax.errorbar(k, mean, yerr=std, marker="o", capsize=3, color="#4c78a8", linewidth=1.5)
        ax.axvline(19, color="#e15759", linestyle="--", linewidth=1.3, label="Selected k = 19")
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("Number of neighbours (k)")
        ax.set_ylabel("Mean cross-validation " + metric.replace("_", " ").title())
        ax.set_xticks(k)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        _save(fig, stem)


def _memory_figure(summary):
    rows = [row for row in summary["memory"] if row["category"] == "major_prediction_workspace"]
    labels = [SHORT_NAMES[row["implementation"]].replace("\n", " ") for row in rows]
    values = [row["mib"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(labels, values, color=[COLORS[row["implementation"]] for row in rows], edgecolor="#444444", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_title("Estimated auxiliary workspace", fontsize=13, pad=12)
    ax.set_xlabel("Major prediction workspace (MiB, logarithmic scale)")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.3f} MiB" for value in values], padding=3, fontsize=9)
    _save(fig, "auxiliary_memory_comparison")


def _rejected_figure(summary):
    rows = summary["rejected_experiments"]
    ratios = [row["candidate_seconds"] / row["reference_seconds"] for row in rows]
    labels = [row["experiment"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    bars = ax.barh(labels, ratios, color="#b0b0b0", edgecolor="#555555", linewidth=0.6)
    ax.axvline(1.0, color="#e15759", linestyle="--", linewidth=1.3, label="Reference = 1.0")
    ax.set_title("Rejected optimization experiments", fontsize=13, pad=12)
    ax.set_xlabel("Candidate runtime / accepted reference runtime")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(bars, labels=[f"{ratio:.2f}x" for ratio in ratios], padding=3, fontsize=8)
    ax.legend(fontsize=9)
    _save(fig, "rejected_optimization_experiments")


def _write_index(entries):
    lines = ["# Figure index", "", "All plots are generated from saved artifacts by `python -m src.benchmarking.generate_figures`.", ""]
    for stem, description, source, section, interpretation in entries:
        lines.extend([
            f"## `{stem}.png` / `{stem}.svg`", "",
            f"- **Shows:** {description}",
            f"- **Data source:** `{source}`",
            f"- **Suggested report section:** {section}",
            f"- **Interpretation:** {interpretation}", "",
        ])
    (FIGURES_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    summary = read_json(BENCHMARK_DIR / "runtime_summary.json")
    _prediction_figures(summary)
    _pipeline_figures(summary)
    _history_figures(summary)
    _phase_figure()
    _cpp_figures(summary)
    _scaling_figure(summary["training_size_scaling"], "train_rows", "Prediction runtime vs training-set size", "runtime_vs_training_size")
    _scaling_figure(summary["query_size_scaling"], "query_rows", "Prediction runtime vs number of queries", "runtime_vs_query_count")
    _quality_figures(summary)
    _cv_figures()
    _memory_figure(summary)
    _rejected_figure(summary)
    entries = [
        ("prediction_runtime_controlled", "controlled 6,000-query median runtime", "raw_prediction_runs.csv", "Prediction-only performance", "C++ has the lowest median; Weka has the highest."),
        ("prediction_runtime_distribution_comprehensive", "all controlled prediction observations and outliers", "raw_prediction_runs.csv", "Stability and outliers", "C++ retains a high implementation-specific outlier."),
        ("prediction_runtime_run_order_comprehensive", "actual interleaved timing sequence", "raw_prediction_runs.csv", "Stability and outliers", "Sequence context separates shared machine state from isolated outliers."),
        ("full_pipeline_runtime_comprehensive", "five-run prepared-input pipeline medians", "raw_full_pipeline_runs.csv", "Full-pipeline performance", "C++ is fastest while Weka includes substantial build and JVM overhead."),
        ("fit_build_runtime", "internal fit/build medians", "raw_full_pipeline_runs.csv", "Full-pipeline performance", "Fit scopes differ and should be compared cautiously."),
        ("custom_optimization_history_linear", "recorded optimization history on a linear scale", "historical_optimization.csv", "Custom optimization history", "Most runtime reduction occurred in early optimization stages."),
        ("custom_optimization_history_log", "recorded optimization history on a log scale", "historical_optimization.csv", "Custom optimization history", "The log view makes later version differences visible."),
        ("speedup_vs_original", "recorded speedup relative to Original", "historical_optimization.csv", "Custom optimization history", "Historical and current measurement regimes are labelled separately."),
        ("speedup_vs_v5_1_comprehensive", "current controlled speedup relative to V5.1", "prediction_runtime_statistics.csv", "Prediction-only performance", "Values above one are faster than accepted V5.1."),
        ("v5_phase_breakdown", "trusted V5 phase medians", "data/processed/v5_local/final_profile.json", "Custom optimization history", "Matrix scoring dominates the measured V5 phases."),
        ("cpp_configuration_screen", "native and portable selection/batch configurations", "raw_cpp_configuration.csv", "C++ configuration experiments", "Native heap/batch 32 is the report configuration."),
        ("runtime_vs_training_size", "runtime over nested training prefixes", "raw_scaling_train.csv", "Training-size scalability", "Observed growth is broadly linear but has machine-state discontinuities."),
        ("runtime_vs_query_count", "runtime over nested query prefixes", "raw_scaling_queries.csv", "Query-count scalability", "Runtime generally rises with query count; Weka includes a fast-state anomaly."),
        ("prediction_throughput", "queries per second at controlled medians", "prediction_runtime_statistics.csv", "Prediction-only performance", "C++ processes the most queries per second."),
        ("milliseconds_per_query", "average median time per query", "prediction_runtime_statistics.csv", "Prediction-only performance", "Per-query cost mirrors the full-pass comparison."),
        ("runtime_vs_f1", "prediction median against final F1", "quality_metrics.csv", "Runtime versus quality", "Quality is nearly unchanged while runtime varies substantially."),
        ("runtime_vs_accuracy", "prediction median against final accuracy", "quality_metrics.csv", "Runtime versus quality", "Tiny accuracy differences should not be overinterpreted."),
        ("confusion_matrix_v5_1", "accepted V5.1 confusion counts", "confusion_counts.csv", "Correctness and quality", "C++ shares this matrix because its outputs are exact."),
        ("cv_f1_by_k_comprehensive", "mean CV F1 with standard-deviation bars", "results/cv_results_summary.csv", "Model selection", "The marked k=19 is the selected parameter."),
        ("cv_balanced_accuracy_by_k", "mean CV balanced accuracy with uncertainty", "results/cv_results_summary.csv", "Model selection", "Balanced accuracy changes little near the selected k."),
        ("implementation_agreement_heatmap", "pairwise class agreement", "correctness_summary.csv", "Correctness and quality", "All pairs exceed 99.96%; C++ and V5.1 are identical."),
        ("auxiliary_memory_comparison", "comparable estimated major workspaces", "memory_storage.csv", "Memory behavior", "The fused C++ heap uses far less algorithm workspace."),
        ("cpp_build_comparison", "portable versus native heap/batch-32 builds", "cpp_configuration_summary.csv", "C++ configuration experiments", "Native AVX2/LTO materially improves this machine's runtime."),
        ("rejected_optimization_experiments", "candidate/reference runtime ratios", "rejected_optimization_experiments.csv", "Rejected experiments", "Every candidate remains rejected under its documented acceptance evidence."),
    ]
    _write_index(entries)
    print(f"Generated {len(entries)} comprehensive figures as PNG and SVG -> {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
