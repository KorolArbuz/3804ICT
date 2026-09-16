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
    svg_path = FIGURES_DIR / f"{stem}.svg"
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _bar(
    values, title, ylabel, stem, labels=None, colors=None,
    value_format="{:.3f}", yerr=None,
):
    labels = labels or [SHORT_NAMES[name] for name in IMPLEMENTATIONS]
    colors = colors or [COLORS[name] for name in IMPLEMENTATIONS]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(
        labels, values, color=colors, edgecolor="#333333", linewidth=0.6,
        yerr=yerr, capsize=4 if yerr is not None else 0,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[value_format.format(value) for value in values], padding=4, fontsize=9)
    _save(fig, stem)


def _prediction_figures(summary):
    statistics = {row["implementation"]: row for row in summary["prediction_only"]}
    medians = {name: statistics[name]["median_seconds"] for name in IMPLEMENTATIONS}
    values = [medians[name] for name in IMPLEMENTATIONS]
    intervals = np.asarray([
        (
            statistics[name]["median_seconds"] - statistics[name]["median_ci95_low"],
            statistics[name]["median_ci95_high"] - statistics[name]["median_seconds"],
        )
        for name in IMPLEMENTATIONS
    ]).T
    _bar(
        values,
        "Controlled prediction runtime with bootstrap 95% intervals",
        "Median prediction time (seconds)",
        "prediction_runtime_controlled",
        yerr=intervals,
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


def _pipeline_figures(summary):
    pipeline = {row["implementation"]: row["median_seconds"] for row in summary["full_pipeline"]}
    _bar(
        [pipeline[name] for name in IMPLEMENTATIONS],
        "Prepared-input implementation pipeline runtime",
        "Median prepared-input pipeline time (seconds)",
        "full_pipeline_runtime_comprehensive",
    )


def _history_figures(summary):
    accepted = [
        row for row in summary["historical_optimization"]
        if row["accepted"] == "true" or row["version"] == "C++ experimental"
    ]
    labels = [row["version"] for row in accepted]
    runtimes = [float(row["median_runtime_seconds"]) for row in accepted]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(labels, runtimes, color="#4c78a8", marker="o", linewidth=1.8)
    ax.scatter(
        labels[-1], runtimes[-1], color=COLORS["custom_cpp"],
        zorder=3, label="Experimental C++",
    )
    ax.set_title(
        "Historical custom KNN timings (mixed benchmark sessions)",
        fontsize=13, pad=12,
    )
    ax.set_ylabel("Recorded prediction time (seconds, logarithmic scale)")
    ax.set_xlabel("Implementation version")
    ax.set_yscale("log")
    ax.set_ylim(min(runtimes) * 0.65, max(runtimes) * 1.7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    _save(fig, "custom_optimization_history_log")

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
    quality = {row["implementation"]: row for row in summary["quality_metrics"]}
    quality["custom_python_v5_1"] = quality["custom"]
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


def _paired_speedup_figure(summary):
    candidates = (
        ("custom_cpp", "C++20 experimental"),
        ("sklearn", "scikit-learn"),
        ("weka", "Weka IBk"),
    )
    grouped = [
        [
            row["speedup_factor"]
            for row in summary["paired_speedups"]
            if row["comparison"] == comparison
        ]
        for name, comparison in candidates
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    plot = ax.boxplot(
        grouped,
        labels=[SHORT_NAMES[name].replace("\n", " ") for name, _ in candidates],
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={
            "marker": "D", "markerfacecolor": "white",
            "markeredgecolor": "black", "markersize": 5,
        },
    )
    for patch, (name, _) in zip(plot["boxes"], candidates):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.78)
    for position, values in enumerate(grouped, start=1):
        ax.scatter([position] * len(values), values, s=16, color="#333333", alpha=0.55)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, label="Equal runtime")
    ax.set_title("Paired trial speedup relative to accepted Python V5.1")
    ax.set_ylabel("Paired speedup vs Python V5.1 (>1 is faster)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    _save(fig, "paired_speedup_distribution")


def _cpp_process_figure():
    path = BENCHMARK_DIR / "cpp_process_mode_diagnostic.csv"
    if not path.exists():
        return False
    rows = read_rows(path)
    modes = ("fresh_process", "persistent_process")
    grouped = [
        [float(row["runtime_seconds"]) for row in rows if row["mode"] == mode]
        for mode in modes
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.boxplot(grouped, labels=["Fresh process", "Persistent process"], showmeans=True)
    for position, values in enumerate(grouped, start=1):
        ax.scatter([position] * len(values), values, s=18, color="#e15759", alpha=0.6)
    ax.set_title("C++ prediction timing by process mode")
    ax.set_ylabel("Internal prediction time (seconds)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "cpp_process_mode_distribution")
    return True


def _session_figure():
    path = BENCHMARK_DIR / "session_summary.csv"
    if not path.exists():
        return False
    rows = read_rows(path)
    session_ids = sorted({row["session_id"] for row in rows})
    x = list(range(len(session_ids)))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for name in IMPLEMENTATIONS:
        values = [
            next(
                float(row["median"])
                for row in rows
                if row["session_id"] == session and row["implementation"] == name
            )
            for session in session_ids
        ]
        ax.plot(
            x, values, marker="o", linewidth=1.6,
            color=COLORS[name], label=DISPLAY_NAMES[name],
        )
    ax.set_xticks(x, session_ids, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Session median prediction time (seconds, log scale)")
    ax.set_title("Prediction runtime by independent benchmark session")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, "runtime_by_session")
    return True


def _affinity_figure():
    path = BENCHMARK_DIR / "affinity_stability_diagnostic.csv"
    if not path.exists():
        return False
    rows = [row for row in read_rows(path) if row["warmup"] == "false"]
    grouped = []
    labels = []
    for mode in ("unpinned", "pinned"):
        for name in ("custom_python_v5_1", "sklearn", "custom_cpp"):
            grouped.append([
                float(row["runtime_seconds"])
                for row in rows
                if row["affinity_mode"] == mode and row["implementation"] == name
            ])
            labels.append(f"{mode.title()}\n{SHORT_NAMES[name].replace(chr(10), ' ')}")
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.boxplot(grouped, labels=labels, showmeans=True)
    ax.set_title("Optional CPU-affinity stability diagnostic")
    ax.set_ylabel("Prediction time (seconds)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "affinity_stability_comparison")
    return True


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
    _scaling_figure(
        summary["training_size_scaling"],
        "train_rows",
        "Prediction runtime vs training-set size",
        "runtime_vs_training_size",
    )
    _scaling_figure(
        summary["query_size_scaling"],
        "query_rows",
        "Prediction runtime vs number of queries",
        "runtime_vs_query_count",
    )
    _quality_figures(summary)
    _cv_figures()
    _memory_figure(summary)
    _rejected_figure(summary)
    _paired_speedup_figure(summary)
    has_process = _cpp_process_figure()
    has_session = _session_figure()
    has_affinity = _affinity_figure()

    entries = [
        ("prediction_runtime_controlled", "controlled medians and deterministic bootstrap 95% intervals", "raw_prediction_runs.csv", "Prediction-only performance", "All raw timings, including outliers, contribute to the intervals."),
        ("prediction_runtime_distribution_comprehensive", "all controlled prediction observations and outliers", "raw_prediction_runs.csv", "Stability and outliers", "C++ retains a high implementation-specific outlier."),
        ("prediction_runtime_run_order_comprehensive", "the actual interleaved timing sequence", "raw_prediction_runs.csv", "Stability and outliers", "Sequence context helps assess shared machine-state effects."),
        ("paired_speedup_distribution", "trial-paired speedup distributions relative to V5.1", "paired_speedups.csv", "Prediction-only performance", "Pairing by trial index preserves shared run context; values above one favor the candidate."),
        ("runtime_vs_training_size", "runtime over nested training prefixes", "raw_scaling_train.csv", "Training-size scalability", "Observed growth is broadly linear over the tested range."),
        ("runtime_vs_query_count", "runtime over nested query prefixes", "raw_scaling_queries.csv", "Query-count scalability", "Runtime generally rises with query count."),
        ("cpp_configuration_screen", "native and portable C++ selection/batch configurations", "raw_cpp_configuration.csv", "C++ configuration experiments", "Native heap/batch 32 is the report configuration."),
        ("custom_optimization_history_log", "recorded optimization history on a log scale", "historical_optimization.csv", "Custom optimization history", "Points come from mixed sessions and are historical context, not a controlled speedup series."),
        ("cv_f1_by_k_comprehensive", "mean CV F1 with standard-deviation bars", "results/cv_results_summary.csv", "Model selection", "The marked k=19 is the selected parameter."),
        ("confusion_matrix_v5_1", "accepted V5.1 confusion counts", "confusion_counts.csv", "Correctness and quality", "C++ shares this matrix because its outputs are exact."),
        ("implementation_agreement_heatmap", "pairwise class agreement", "correctness_summary.csv", "Correctness and quality", "All pairs exceed 99.96%; C++ and V5.1 are identical."),
        ("full_pipeline_runtime_comprehensive", "five-run prepared-input implementation pipeline medians", "raw_full_pipeline_runs.csv", "Prepared-input pipeline performance", "C++ is fastest while Weka includes substantial build and JVM overhead."),
        ("prediction_throughput", "queries per second at controlled medians", "prediction_runtime_statistics.csv", "Prediction-only performance", "C++ processes the most queries per second."),
        ("v5_phase_breakdown", "trusted V5 phase medians", "data/processed/v5_local/final_profile.json", "Custom optimization history", "Matrix scoring dominates the measured V5 phases."),
        ("auxiliary_memory_comparison", "comparable estimated major workspaces", "raw_memory.csv", "Memory behavior", "The fused C++ heap uses less algorithm workspace."),
        ("rejected_optimization_experiments", "candidate/reference runtime ratios", "rejected_optimization_experiments.csv", "Rejected experiments", "Every candidate remains rejected under its recorded evidence."),
    ]
    if has_process:
        entries.append(
            ("cpp_process_mode_distribution", "20 fresh-process and 20 persistent-process C++ timings", "cpp_process_mode_diagnostic.csv", "Runtime stability", "The diagnostic tests whether process reuse explains C++ variability.")
        )
    if has_session:
        entries.append(
            ("runtime_by_session", "median runtime for each completed real session", "session_summary.csv", "Cross-session reproducibility", "More independent sessions are needed before making a reproducibility claim.")
        )
    if has_affinity:
        entries.append(
            ("affinity_stability_comparison", "explicit-CPU pinned and unpinned timing distributions", "affinity_stability_diagnostic.csv", "Runtime stability", "This optional diagnostic appears only when actual measurements exist.")
        )
    _write_index(entries)
    print(f"Generated {len(entries)} compact figures as PNG and SVG -> {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
