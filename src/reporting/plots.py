from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import RESULTS_DIR
from src.common.utils import read_json


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    pyplot().close(figure)


def plot_cv(summary, selected_k, output):
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.errorbar(
        summary.k,
        summary.f1_mean,
        yerr=summary.f1_std,
        marker="o",
        capsize=3,
        color="#2266aa",
        label="Mean validation F1 +/- one fold SD",
    )
    axis.axvline(
        selected_k,
        color="#bd5328",
        linestyle="--",
        label=f"Selected k = {selected_k}",
    )
    axis.set(
        xlabel="Number of neighbors (k)",
        ylabel="F1 for default (class 1)",
        title="Training-only stratified 5-fold cross-validation",
        ylim=(0, 1),
        xticks=summary.k,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    save(figure, output)


def plot_results(results_dir=RESULTS_DIR, output_dir=None):
    from src.evaluation.comparison import DISPLAY

    results_dir = Path(results_dir)
    output_dir = Path(output_dir or results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = read_json(results_dir / "selected_parameters.json")
    plt = pyplot()

    cv_summary = pd.read_csv(results_dir / "cv_results_summary.csv")
    plot_cv(
        cv_summary,
        selected["selected_k"],
        output_dir / "cv_f1_vs_k.png",
    )

    metrics = pd.read_csv(results_dir / "metrics_comparison.csv", na_values=["undefined"])
    columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
    ]
    column_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Balanced\naccuracy",
        "ROC-AUC",
        "Average\nprecision",
    ]

    figure, axis = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(columns))
    width = 0.8 / len(metrics)
    colors = ["#2266aa", "#e69f00", "#009e73"]

    for index, row in enumerate(metrics.itertuples(index=False)):
        values = [getattr(row, column) for column in columns]
        bar_positions = positions + (index - (len(metrics) - 1) / 2) * width
        bars = axis.bar(
            bar_positions,
            values,
            width,
            label=DISPLAY[row.implementation],
            color=colors[index],
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=3)

    axis.set(
        xticks=positions,
        xticklabels=column_labels,
        ylim=(0, 1.08),
        ylabel="Test-set score",
        title="KNN implementations on the same untouched test set",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)
    save(figure, output_dir / "metrics_comparison.png")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    implementation_labels = [DISPLAY[name] for name in metrics.implementation]
    bars = axis.bar(
        implementation_labels,
        metrics.prediction_seconds,
        color=colors[:len(metrics)],
    )
    axis.bar_label(bars, fmt="%.3f s", padding=4)
    axis.set(
        ylabel="Measured prediction time (seconds)",
        title="One complete prediction pass; loading and build excluded",
        ylim=(0, metrics.prediction_seconds.max() * 1.2),
    )
    save(figure, output_dir / "runtime_comparison.png")

    for row in metrics.itertuples(index=False):
        if row.implementation != "custom":
            continue
        matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
        figure, axis = plt.subplots(figsize=(5, 4.5))
        rendered = axis.imshow(matrix, cmap="Blues", vmin=0)

        for (i, j), value in np.ndenumerate(matrix):
            text_color = "white" if value > matrix.max() / 2 else "black"
            axis.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
            )

        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["0: no default", "1: default"],
            yticklabels=["0: no default", "1: default"],
            xlabel="Predicted class",
            ylabel="True class",
            title=DISPLAY[row.implementation],
        )
        figure.colorbar(rendered, ax=axis, label="Test samples")
        save(figure, output_dir / f"confusion_matrix_{row.implementation}.png")

    print(f"Generated 4 measured-result charts -> {output_dir}")
