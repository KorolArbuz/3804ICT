"""Generate publication-quality runtime figures from raw benchmark CSV files."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .benchmark_utils import (
    BENCHMARK_DIR,
    COLORS,
    DISPLAY_NAMES,
    FIGURES_DIR,
    IMPLEMENTATIONS,
    read_rows,
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
    })
    return plt


def _save(figure, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(base_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    _pyplot().close(figure)


def _prediction_values(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["is_warmup"].lower() == "false":
            grouped[row["implementation"]].append(float(row["prediction_seconds"]))
    return grouped


def _bar_chart(values, title, ylabel, output, value_format="{:.3f} s"):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    labels = [DISPLAY_NAMES[name] for name in IMPLEMENTATIONS]
    heights = [values[name] for name in IMPLEMENTATIONS]
    bars = axis.bar(
        np.arange(len(labels)),
        heights,
        color=[COLORS[name] for name in IMPLEMENTATIONS],
        width=0.68,
    )
    axis.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        ylabel=ylabel,
        title=title,
        ylim=(0, max(heights) * 1.18),
    )
    axis.grid(axis="y", alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, heights):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(heights) * 0.025,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    _save(figure, output)


def plot_prediction_bar(grouped, output_dir):
    medians = {name: float(np.median(grouped[name])) for name in IMPLEMENTATIONS}
    _bar_chart(
        medians,
        "One complete prediction pass; loading and build excluded",
        "Measured prediction time (seconds)",
        output_dir / "prediction_bar_chart",
    )


def plot_prediction_boxplot(grouped, output_dir):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    data = [grouped[name] for name in IMPLEMENTATIONS]
    box = axis.boxplot(
        data,
        labels=[DISPLAY_NAMES[name] for name in IMPLEMENTATIONS],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 5},
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.65},
    )
    for patch, name in zip(box["boxes"], IMPLEMENTATIONS):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.78)
    axis.set(
        ylabel="Measured prediction time (seconds, logarithmic scale)",
        title="Prediction-time distribution across all retained runs",
    )
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    _save(figure, output_dir / "prediction_boxplot")


def plot_full_pipeline(rows, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["implementation"]].append(float(row["total_seconds"]))
    medians = {name: float(np.median(grouped[name])) for name in IMPLEMENTATIONS}
    _bar_chart(
        medians,
        "Full experiment runtime from prepared inputs",
        "End-to-end runtime (seconds)",
        output_dir / "full_pipeline_bar_chart",
    )


def _historical_rows(path, cpp_median):
    label_map = {
        "baseline_original": "Original",
        "optimized_v1": "V1",
        "optimized_v2_single_thread": "V2",
        "optimized_v3_single_thread": "V3",
        "optimized_v4_threshold_prefilter": "V4",
        "optimized_v5_single_thread": "V5",
        "optimized_v5_1_single_thread": "V5.1",
    }
    values = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["version"] in label_map and row["prediction_seconds_summary"]:
                values.append((
                    label_map[row["version"]],
                    float(row["prediction_seconds_summary"]),
                ))
    values.append(("C++\nexperimental", cpp_median))
    return values


def plot_optimization_timeline(history_path, cpp_median, output_dir):
    plt = _pyplot()
    history = _historical_rows(history_path, cpp_median)
    labels = [item[0] for item in history]
    values = [item[1] for item in history]
    original = values[0]
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    positions = np.arange(len(labels))
    axis.plot(positions, values, marker="o", linewidth=1.8, color="#4c78a8")
    axis.scatter(positions[-1], values[-1], color=COLORS["custom_cpp"], s=62, zorder=3)
    axis.set_yscale("log")
    axis.set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Prediction time (seconds, logarithmic scale)",
        title="Historical custom KNN optimization timeline",
    )
    axis.grid(axis="y", which="both", alpha=0.22)
    for position, value in zip(positions, values):
        axis.annotate(
            f"{value:.3f}s\n{original / value:.1f}x",
            (position, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.3,
        )
    _save(figure, output_dir / "custom_optimization_timeline")


def plot_relative_speedup(grouped, output_dir):
    medians = {name: float(np.median(grouped[name])) for name in IMPLEMENTATIONS}
    baseline = medians["custom_python_v5_1"]
    speedups = {name: baseline / medians[name] for name in IMPLEMENTATIONS}
    _bar_chart(
        speedups,
        "Prediction speedup relative to accepted Python V5.1",
        "Speedup factor (V5.1 = 1.00x; higher is faster)",
        output_dir / "relative_speedup_chart",
        value_format="{:.2f}x",
    )


def plot_run_order_drift(rows, output_dir):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10.2, 5.5))
    for name in IMPLEMENTATIONS:
        implementation_rows = [
            row
            for row in rows
            if row["implementation"] == name and row["is_warmup"].lower() == "false"
        ]
        axis.plot(
            [int(row["sequence_index"]) for row in implementation_rows],
            [float(row["prediction_seconds"]) for row in implementation_rows],
            marker="o",
            markersize=4,
            linewidth=1.2,
            color=COLORS[name],
            label=DISPLAY_NAMES[name],
        )
    axis.set(
        xlabel="Global timed execution order (interleaved)",
        ylabel="Prediction time (seconds, logarithmic scale)",
        title="Run-order drift with every raw timed result retained",
    )
    axis.set_yscale("log")
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    axis.set_axisbelow(True)
    _save(figure, output_dir / "run_order_drift")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", type=Path, default=BENCHMARK_DIR / "raw_prediction_benchmark.csv"
    )
    parser.add_argument(
        "--full-pipeline",
        type=Path,
        default=BENCHMARK_DIR / "raw_full_pipeline_benchmark.csv",
    )
    parser.add_argument(
        "--history", type=Path, default=Path("results/custom_knn_optimization.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    prediction_rows = read_rows(args.prediction)
    full_rows = read_rows(args.full_pipeline)
    grouped = _prediction_values(prediction_rows)
    missing = [name for name in IMPLEMENTATIONS if len(grouped[name]) < 15]
    if missing:
        raise ValueError(f"Prediction benchmark lacks 15 timed rows for: {missing}")

    plot_prediction_bar(grouped, args.output_dir)
    plot_prediction_boxplot(grouped, args.output_dir)
    plot_full_pipeline(full_rows, args.output_dir)
    plot_optimization_timeline(
        args.history,
        float(np.median(grouped["custom_cpp"])),
        args.output_dir,
    )
    plot_relative_speedup(grouped, args.output_dir)
    plot_run_order_drift(prediction_rows, args.output_dir)
    print(f"Generated 6 figures as PNG and SVG -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
