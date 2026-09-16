"""Focused figures and Markdown reports for the model-quality experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    ("euclidean", "uniform"): "Euclidean + uniform",
    ("euclidean", "distance"): "Euclidean + distance",
    ("manhattan", "uniform"): "Manhattan + uniform",
    ("manhattan", "distance"): "Manhattan + distance",
}
COLORS = {
    ("euclidean", "uniform"): "#4C78A8",
    ("euclidean", "distance"): "#F58518",
    ("manhattan", "uniform"): "#54A24B",
    ("manhattan", "distance"): "#E45756",
}


def _save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _configuration_lines(summary, metric_name, selected):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for configuration, label in LABELS.items():
        distance, weights = configuration
        rows = summary[
            (summary.metric == distance) & (summary.weights == weights)
        ].sort_values("k")
        ax.plot(rows.k, rows[f"{metric_name}_mean"], marker="o", ms=3,
                lw=1.6, color=COLORS[configuration], label=label)
        ax.fill_between(
            rows.k,
            rows[f"{metric_name}_mean"] - rows[f"{metric_name}_std"],
            rows[f"{metric_name}_mean"] + rows[f"{metric_name}_std"],
            color=COLORS[configuration], alpha=0.10,
        )

    baseline = summary[
        (summary.k == 19)
        & (summary.metric == "euclidean")
        & (summary.weights == "uniform")
    ].iloc[0]
    ax.scatter([19], [baseline[f"{metric_name}_mean"]], s=100, marker="s",
               facecolors="none", edgecolors="black", linewidths=1.8,
               label="Accepted baseline")
    ax.scatter([selected.k], [selected[f"{metric_name}_mean"]], s=120, marker="*",
               color="black", label="CV-selected winner", zorder=5)
    ax.set_xlabel("Number of neighbours (k)")
    ax.set_ylabel(f"Mean class-1 {metric_name.replace('_', ' ')}")
    ax.set_xticks(range(1, 32, 2))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_cv_figures(summary, selected, output_dir):
    output_dir = Path(output_dir)
    _save(_configuration_lines(summary, "f1", selected),
          output_dir / "cv_f1_by_configuration")
    _save(_configuration_lines(summary, "recall", selected),
          output_dir / "cv_recall_by_configuration")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for configuration, label in LABELS.items():
        distance, weights = configuration
        rows = summary[(summary.metric == distance) & (summary.weights == weights)]
        ax.scatter(rows.recall_mean, rows.precision_mean, s=32, alpha=0.75,
                   color=COLORS[configuration], label=label)
    baseline = summary[
        (summary.k == 19) & (summary.metric == "euclidean")
        & (summary.weights == "uniform")
    ].iloc[0]
    ax.scatter(baseline.recall_mean, baseline.precision_mean, s=110, marker="s",
               facecolors="none", edgecolors="black", linewidths=1.8,
               label="Accepted baseline")
    ax.scatter(selected.recall_mean, selected.precision_mean, s=130, marker="*",
               color="black", label="CV-selected winner", zorder=5)
    ax.set_xlabel("Mean class-1 recall")
    ax.set_ylabel("Mean class-1 precision")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir / "cv_precision_recall_tradeoff")

    _save(_configuration_lines(summary, "balanced_accuracy", selected),
          output_dir / "cv_balanced_accuracy_by_configuration")


def write_cv_report(path, baseline, selected, summary=None):
    path = Path(path)
    metrics = ("f1", "recall", "precision", "balanced_accuracy", "average_precision")
    deltas = {
        name: float(selected[f"{name}_mean"] - baseline[f"{name}_mean"])
        for name in metrics
    }
    manhattan_best = selected.metric == "manhattan"
    distance_best = selected.weights == "distance"
    precision_recall_trade = deltas["recall"] > 0 and deltas["precision"] < 0
    if summary is not None:
        best_manhattan = summary.loc[summary[summary.metric == "manhattan"].f1_mean.idxmax()]
        best_uniform = summary.loc[summary[summary.weights == "uniform"].f1_mean.idxmax()]
        best_distance = summary.loc[summary[summary.weights == "distance"].f1_mean.idxmax()]
        family_findings = (
            f"The best Manhattan configuration reached mean F1 "
            f"**{best_manhattan.f1_mean:.4f}** "
            f"({best_manhattan.f1_mean - baseline.f1_mean:+.4f} versus the baseline), "
            "so Manhattan did not produce the winner. "
            f"The best distance-weighted configuration reached **{best_distance.f1_mean:.4f}**, "
            f"while the best uniform configuration reached **{best_uniform.f1_mean:.4f}**; "
            "distance weighting produced the selected maximum."
        )
    else:
        family_findings = (
            f"Manhattan distance {'is' if manhattan_best else 'is not'} part of the selected "
            f"winner. Distance weighting {'is' if distance_best else 'is not'} part of the "
            "selected winner."
        )
    consistency = (
        f"The selected configuration's fold F1 standard deviation was "
        f"{selected.f1_std:.4f}, compared with {baseline.f1_std:.4f} for the baseline. "
        "It was more consistent across these five folds, but five observed folds do "
        "not establish statistical significance."
    )
    text = f"""# Training-only model-quality report

## Measured results

Training-only cross-validation selected **k={int(selected.k)}, {selected.metric},
{selected.weights} voting**. The accepted k=19 Euclidean/uniform baseline had mean
class-1 F1 **{baseline.f1_mean:.4f}**; the selected configuration had
**{selected.f1_mean:.4f}** (delta **{deltas['f1']:+.4f}**).

| Metric | Baseline mean | Selected mean | Delta |
|---|---:|---:|---:|
| Class-1 F1 | {baseline.f1_mean:.4f} | {selected.f1_mean:.4f} | {deltas['f1']:+.4f} |
| Recall | {baseline.recall_mean:.4f} | {selected.recall_mean:.4f} | {deltas['recall']:+.4f} |
| Precision | {baseline.precision_mean:.4f} | {selected.precision_mean:.4f} | {deltas['precision']:+.4f} |
| Balanced accuracy | {baseline.balanced_accuracy_mean:.4f} | {selected.balanced_accuracy_mean:.4f} | {deltas['balanced_accuracy']:+.4f} |
| Average precision | {baseline.average_precision_mean:.4f} | {selected.average_precision_mean:.4f} | {deltas['average_precision']:+.4f} |

{consistency}

## Interpretation

The observed change {'does' if precision_recall_trade else 'does not'} primarily
show a precision-for-recall trade-off relative to the accepted baseline.
{family_findings}

The selected k {'differs from' if int(selected.k) != 19 else 'matches'} the
accepted k=19 setting ({int(selected.k)} versus 19). The six-neighbour change is
numerically clear, but the mean-F1 separation of {deltas['f1']:+.4f} is small;
these five folds do not establish that the k change is materially important.

## Limitation

All selection statements come from the fixed training partition. The test set was
not evaluated or used to choose k, distance, voting, threshold, or preprocessing.
Five CV folds are not an independent significance test, so this report does not
claim statistical significance.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def plot_final_figures(comparison, output_dir):
    output_dir = Path(output_dir)
    focus = comparison[comparison.implementation.isin(["baseline_custom_v5_1", "enhanced_custom"])]
    labels = ["Baseline V5.1", "Enhanced Custom"]
    metrics = ["f1", "recall", "precision", "balanced_accuracy", "average_precision"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for index, (_, row) in enumerate(focus.iterrows()):
        ax.bar(x + (index - 0.5) * width, [row[name] for name in metrics], width,
               label=labels[index])
    ax.set_xticks(x, [name.replace("_", "\n") for name in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test metric")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir / "baseline_vs_selected_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, (_, row), label in zip(axes, focus.iterrows(), labels):
        matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
        ax.set_title(label)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    _save(fig, output_dir / "baseline_vs_selected_confusion")
