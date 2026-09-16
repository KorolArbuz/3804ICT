"""Reports and focused figures for the frozen V2 operating-point study."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .operating_point import LEGACY_NOTICE, POINT_NAMES


METRIC_COLUMNS = (
    "threshold", "FP", "FN", "TP", "TN", "accuracy", "precision", "recall",
    "f1", "f0_5", "balanced_accuracy", "predicted_positive_rate",
)


def _save(fig, output_dir, name):
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            clean = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
            path.write_text(clean + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def _points_frame(points):
    return pd.DataFrame([{"operating_point": name, **points[name].to_dict()} for name in POINT_NAMES])


def _point_table(points):
    frame = _points_frame(points)
    return frame[["operating_point", *METRIC_COLUMNS]].to_markdown(index=False, floatfmt=".6f")


def _delta_table(points):
    frame = _points_frame(points).set_index("operating_point")
    reference = frame.loc["current_v2_threshold"]
    rows = []
    for name in POINT_NAMES:
        row = frame.loc[name]
        values = {"operating_point": name}
        for metric in ("FP", "FN", "TP", "TN", "accuracy", "precision", "recall", "f1", "f0_5"):
            values[f"delta_{metric}"] = row[metric] - reference[metric]
        rows.append(values)
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt="+.6f")


def create_figures(output_dir, frontier, points):
    output_dir = Path(output_dir)
    selected = _points_frame(points)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.scatter(frontier.FP, frontier.recall, s=6, alpha=0.12, color="slategray",
               label="Exact OOF states", rasterized=True)
    ax.scatter(selected.FP, selected.recall, s=55, color="crimson", zorder=3)
    for row in selected.itertuples():
        ax.annotate(row.operating_point, (row.FP, row.recall), xytext=(4, 4),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("False positives (training OOF)")
    ax.set_ylabel("Recall")
    ax.set_title("False positives versus recall")
    ax.grid(alpha=0.2)
    _save(fig, output_dir, "fp_vs_recall")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sizes = 8 + 90 * frontier.FP / max(float(frontier.FP.max()), 1.0)
    ax.scatter(frontier.recall, frontier.precision, s=sizes, alpha=0.10, color="teal",
               rasterized=True)
    ax.scatter(selected.recall, selected.precision,
               s=35 + 100 * selected.FP / max(float(frontier.FP.max()), 1.0),
               color="darkorange", edgecolor="black", linewidth=0.4, zorder=3)
    for row in selected.itertuples():
        ax.annotate(f"{row.operating_point}\nFP={row.FP}", (row.recall, row.precision),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall frontier; marker area reflects FP")
    ax.grid(alpha=0.2)
    _save(fig, output_dir, "precision_recall_fp_frontier")

    visible = frontier[frontier.threshold <= 1.0]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for metric, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                          ("f0_5", "F0.5"), ("accuracy", "Accuracy")):
        ax.plot(visible.threshold, visible[metric], linewidth=1.3, label=label)
    for row in selected.itertuples():
        ax.axvline(row.threshold, color="black", alpha=0.16, linewidth=0.8)
    ax.set_xlabel("Threshold (predict class 1 when score ≥ threshold)")
    ax.set_ylabel("Training OOF metric")
    ax.set_title("Threshold-dependent operating metrics")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    _save(fig, output_dir, "threshold_metrics")


def create_oof_outputs(output_dir, frontier, points, current_threshold):
    output_dir = Path(output_dir)
    create_figures(output_dir, frontier, points)
    selected = _points_frame(points)
    coincidences = [", ".join(group.operating_point) for _, group in selected.groupby("threshold") if len(group) > 1]
    coincidence_text = (
        "The independently predeclared objectives that converged to one numerical threshold were: "
        + "; ".join(coincidences) + "."
        if coincidences else "All predeclared objectives produced distinct numerical thresholds."
    )
    report = f"""# Frozen V2 operating points: training-only OOF evidence

The frozen k=101 Model Quality V2 score model was evaluated with five-fold stratified OOF prediction inside the original 24,000-row training partition. Every row was scored once by a model that did not train on it. The legacy test was not used to choose a constraint, objective, candidate threshold, or tie-break.

The decision rule is `class 1 if probability_class_1 >= threshold`. The current reference threshold is {current_threshold:.12f} and was read from the already-frozen final V2 configuration rather than reselected.

## Frozen operating points

{_point_table(points)}

{coincidence_text}

ROC-AUC and Average Precision describe the shared continuous score ranking and therefore do not vary by threshold.

## Deltas versus the current V2 threshold

{_delta_table(points)}

Fewer false positives usually come with fewer true positives and lower recall. Each constrained operating point is preferred only when its stated objective and constraints match the intended use. None is universally superior.
"""
    (output_dir / "oof_operating_point_report.md").write_text(report, encoding="utf-8", newline="\n")


def _legacy_table(comparison):
    columns = (
        "operating_point", "threshold", "accuracy", "precision", "recall", "specificity",
        "f1", "f0_5", "balanced_accuracy", "TN", "FP", "FN", "TP",
    )
    return comparison[list(columns)].to_markdown(index=False, floatfmt=".6f")


def _tradeoff_table(comparison):
    columns = (
        "operating_point", "FP_avoided", "TP_lost", "additional_FN", "TN_gained",
        "net_correct_classification_change", "FP_avoided_per_TP_lost",
        "TP_lost_per_100_FP_avoided", "accuracy_delta", "precision_delta",
        "recall_delta", "f1_delta", "f0_5_delta",
    )
    return comparison[list(columns)].to_markdown(index=False, floatfmt=".6f")


def _bootstrap_table(bootstrap):
    return bootstrap[[
        "operating_point", "metric", "observed_difference", "ci_lower_95",
        "ci_upper_95", "includes_zero",
    ]].to_markdown(index=False, floatfmt=".6f")


def create_final_report(output_dir, selection, comparison, shared, parity, bootstrap):
    output_dir = Path(output_dir)
    candidates = comparison[comparison.operating_point != "current_v2_threshold"]
    largest_reduction = candidates.sort_values(
        ["FP_avoided", "recall", "f1"], ascending=[False, False, False], kind="stable"
    ).iloc[0]
    precision_first = comparison[comparison.operating_point == "max_precision_recall_50"].iloc[0]
    balanced = comparison[comparison.operating_point == "min_fp_accuracy_guard"].iloc[0]
    parity_rows = []
    for name in POINT_NAMES:
        item = parity["operating_points"][name]
        parity_rows.append({"operating_point": name, "agreement": item["agreement_count"],
                            "disagreements": item["disagreement_count"]})
    parity_table = pd.DataFrame(parity_rows).to_markdown(index=False)
    selected_thresholds = pd.DataFrame([
        {"operating_point": name, "threshold": selection["operating_points"][name]["threshold"]}
        for name in POINT_NAMES
    ])
    coincidences = [", ".join(group.operating_point) for _, group in selected_thresholds.groupby("threshold") if len(group) > 1]
    coincidence_text = (
        "The predeclared objectives that converged to the same frozen threshold were: "
        + "; ".join(coincidences) + "."
        if coincidences else "Every predeclared objective produced a distinct threshold."
    )
    report = f"""# Final V2 operating-point optimization

## 1. Objective

This study asks whether the frozen final V2 KNN can operate with fewer false positives while preserving predeclared minimum recall, F1, or accuracy. It changes only the score threshold; it does not change the neighbour model or score ranking.

## 2. Frozen V2 model

The model remains k=101, Euclidean, inverse-distance voting, signed-log money features, structured PAY representation, Blocks A+B, and PAY/delinquency weight 1.0. Its configuration hash was verified before OOF scoring and again before legacy evaluation.

## 3. Training-only OOF protocol

Five-fold `StratifiedKFold(shuffle=True, random_state=42)` generated one held-out score for each of 24,000 training rows. Every transform was fitted inside its fold. Threshold objectives, constraints, and tie-breaks were predeclared, and all six thresholds were frozen before the legacy test was scored.

## 4. Threshold frontier

The exact frontier contains {selection['threshold_candidate_count']} threshold candidates, including every unique OOF score, fixed boundary thresholds, and the frozen current threshold. The three-dimensional FP/recall/F1 Pareto set contains {selection['pareto_3d_size']} points; the min-FP-by-recall frontier contains {selection['frontier_2d_size']} points. Shared OOF score ranking was AP {selection['oof_ranking_metrics']['average_precision']:.6f} and ROC-AUC {selection['oof_ranking_metrics']['roc_auc']:.6f}.

## 5. Frozen operating points

See `operating_points.json` and `oof_operating_point_report.md` for the immutable training-OOF thresholds and metrics. No threshold was selected from legacy labels. {coincidence_text}

## 6. OOF trade-offs

The OOF tables show that FP reduction generally exchanges true positives for true negatives. A lower-FP operating point is appropriate only for the constraint that defined it; it is not a universal improvement.

## 7. Legacy comparison

**{LEGACY_NOTICE}** The same deterministic manual-kernel continuous score vector was used for every threshold. Its shared AP was {shared['average_precision']:.6f} and ROC-AUC was {shared['roc_auc']:.6f}.

{_legacy_table(comparison)}

## 8. FP avoided versus TP lost

{_tradeoff_table(comparison)}

The largest observed FP reduction came from `{largest_reduction.operating_point}`: {int(largest_reduction.FP_avoided)} FP avoided for {int(largest_reduction.TP_lost)} TP lost, or {largest_reduction.TP_lost_per_100_FP_avoided:.3f} TP lost per 100 FP avoided. This ratio is descriptive and is not a business-cost model.

## 9. Paired bootstrap

Intervals below are deterministic 95% percentile intervals from 10,000 paired row resamples. Differences are candidate minus current threshold; `includes_zero` is descriptive and is not automatically labelled statistical significance.

{_bootstrap_table(bootstrap)}

## 10. Manual versus sklearn parity

All {parity['rows']} legacy rows were scored independently by `CustomV2KNNClassifier` and scikit-learn. Maximum absolute probability difference was {parity['maximum_absolute_probability_difference']:.3g}; {parity['probability_differences_gt_1e_10']} rows differed by more than 1e-10.

{parity_table}

## 11. Limitations

The legacy test has prior analyst exposure and provides comparability rather than independent validation. Bootstrap intervals characterize this fixed legacy sample. Threshold constraints encode technical trade-offs rather than monetary costs. Results apply to the frozen V2 scores and observed class prevalence. No new independent data was collected.

## 12. Conclusion

False positives can be reduced materially under the predeclared OOF constraints, but the reduction costs true positives. `{largest_reduction.operating_point}` produced the largest observed reduction: {int(largest_reduction.FP_avoided)} fewer FP, recall change {largest_reduction.recall_delta:+.6f}, F1 change {largest_reduction.f1_delta:+.6f}, and {largest_reduction.TP_lost_per_100_FP_avoided:.3f} TP lost per 100 FP avoided. Among the recall/F1/accuracy floor constraints, `min_fp_f1_52` removed the most FP. The accuracy-guard point stayed within its declared OOF guard by construction and changed legacy accuracy by {balanced.accuracy_delta:+.6f}; its constraint was not reopened. The F0.5 point is the most conservative toward false-positive avoidance. The coincident recall-constrained and precision-first point is the most balanced observed lower-FP trade-off because it retained more sensitivity (legacy precision {precision_first.precision:.6f}, recall {precision_first.recall:.6f}) and changed F1 by only {precision_first.f1_delta:+.6f}. The accuracy-guard point remains the explicit multi-floor alternative. These descriptions name their objectives; none is universally best.
"""
    (output_dir / "operating_point_report.md").write_text(report, encoding="utf-8", newline="\n")
