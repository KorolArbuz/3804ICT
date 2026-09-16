"""Reports and training-OOF figures for the frozen-model threshold study.

The OOF writer accepts no test results. Test reporting is a separate operation
called only after the threshold selection has been frozen and evaluated.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THRESHOLD_NAMES = (
    "default_threshold",
    "f1_optimized_threshold",
    "recall_oriented_threshold",
)
LABELS = {
    "default_threshold": "Selected model, threshold 0.5",
    "f1_optimized_threshold": "Selected model, F1 threshold",
    "recall_oriented_threshold": "Selected model, recall threshold",
}
SHORT_LABELS = {
    "default_threshold": "Default 0.5",
    "f1_optimized_threshold": "F1 optimized",
    "recall_oriented_threshold": "Recall oriented",
}
COLORS = {
    "default_threshold": "#4C78A8",
    "f1_optimized_threshold": "#E45756",
    "recall_oriented_threshold": "#54A24B",
}
MARKERS = {
    "default_threshold": "o",
    "f1_optimized_threshold": "D",
    "recall_oriented_threshold": "*",
}


def _records(rows):
    """Accept either a DataFrame or a list of metric dictionaries."""
    if hasattr(rows, "to_dict"):
        return rows.to_dict(orient="records")
    return list(rows)


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _threshold(value):
    return f"{float(value):.12g}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _metric_tables(rows, baseline=None):
    performance, counts = [], []
    if baseline is not None:
        labelled = [("Accepted V5.1: k=19, uniform, argmax", baseline, "argmax")]
    else:
        labelled = []
    labelled.extend(
        (LABELS[row["threshold_name"]], row, _threshold(row["threshold"]))
        for row in rows
    )
    for label, row, threshold in labelled:
        performance.append([
            label, threshold,
            *[f"{float(row[name]):.6f}" for name in (
                "accuracy", "precision", "recall", "specificity", "f1",
                "balanced_accuracy",
            )],
        ])
        count = sum(int(row[name]) for name in ("TN", "FP", "FN", "TP"))
        positive_rate = row.get("predicted_positive_rate", (row["TP"] + row["FP"]) / count)
        counts.append([
            label, *[int(row[name]) for name in ("TN", "FP", "FN", "TP")],
            f"{float(positive_rate):.6f}",
        ])
    return (
        _table(
            ["Rule", "Threshold", "Accuracy", "Precision", "Recall", "Specificity", "F1",
             "Balanced accuracy"],
            performance,
        ) + "\n\n" + _table(
            ["Rule", "TN", "FP", "FN", "TP", "Predicted-positive rate"], counts,
        )
    )


def _changes(rows):
    reference = next(row for row in rows if row["threshold_name"] == "default_threshold")
    changes = []
    for row in rows:
        if row["threshold_name"] == "default_threshold":
            continue
        changes.append([
            SHORT_LABELS[row["threshold_name"]],
            *[f"{float(row[name]) - float(reference[name]):+.6f}" for name in (
                "f1", "recall", "precision", "balanced_accuracy", "accuracy",
            )],
            f"{int(row['TP']) - int(reference['TP']):+d}",
            f"{int(reference['FN']) - int(row['FN']):+d}",
            f"{int(row['FP']) - int(reference['FP']):+d}",
        ])
    return _table(
        ["Rule vs 0.5", "F1 delta", "Recall delta", "Precision delta", "Balanced accuracy delta",
         "Accuracy delta", "Extra TP", "FN avoided", "Extra FP"], changes,
    )


def _tradeoffs(rows):
    default = next(row for row in rows if row["threshold_name"] == "default_threshold")
    paragraphs = []
    for row in rows:
        if row["threshold_name"] == "default_threshold":
            continue
        recall_delta = float(row["recall"]) - float(default["recall"])
        precision_delta = float(row["precision"]) - float(default["precision"])
        f1_delta = float(row["f1"]) - float(default["f1"])
        tp_delta = int(row["TP"]) - int(default["TP"])
        fp_delta = int(row["FP"]) - int(default["FP"])
        fn_avoided = int(default["FN"]) - int(row["FN"])
        paragraphs.append(
            f"**{SHORT_LABELS[row['threshold_name']]}:** F1 "
            f"{'rose' if f1_delta > 0 else 'fell' if f1_delta < 0 else 'was unchanged'} "
            f"({f1_delta:+.6f}); recall changed by **{100 * recall_delta:+.2f} percentage "
            f"points**, and precision changed by **{100 * precision_delta:+.2f} percentage "
            f"points**. This detected **{tp_delta:+d}** additional defaults, avoided "
            f"**{fn_avoided:+d}** false negatives, and introduced **{fp_delta:+d}** "
            "additional false positives relative to threshold 0.5."
        )
    return "\n\n".join(paragraphs)


def _selection_description(selection):
    lines = [
        f"The default threshold is **{_threshold(selection['default_threshold'])}**.",
        f"The F1-optimized threshold is **{_threshold(selection['f1_optimized_threshold'])}**.",
    ]
    recall = selection.get("recall_oriented_threshold")
    if recall is None:
        lines.append(
            "No recall-oriented threshold met the predeclared OOF precision floor "
            f"of **{float(selection['precision_floor']):.2f}**; that operating point is unavailable."
        )
    else:
        lines.append(
            f"The recall-oriented threshold is **{_threshold(recall)}**, selected subject "
            f"to OOF precision >= **{float(selection['precision_floor']):.2f}**."
        )
    lines.append(
        "The rule is **predict class 1 when probability_class_1 >= threshold**. "
        "A score equal to the threshold is always classified as class 1."
    )
    return "\n\n".join(lines)


def _methodology(selection):
    return (
        f"The fixed training partition supplied **{int(selection['oof_rows']):,}** "
        "out-of-fold (OOF) class-1 scores. Each training row received exactly one score "
        "from a model that did not train on it. Five-fold StratifiedKFold used shuffle=True "
        "and random_state=42, retaining the existing seed-42 80/20 split. Preprocessing "
        "was fitted separately on each fold's training rows and then applied to that "
        "fold's validation rows. The score generator was scikit-learn "
        "KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance', "
        "algorithm='brute', n_jobs=1).\n\n"
        f"The deterministic threshold grid contains **{int(selection['candidate_count']):,}** "
        "candidates: all unique OOF-score breakpoints plus 0, 0.5, 1, and "
        "nextafter(1, +infinity). The last value is the next representable float "
        "above 1 and represents the all-negative decision even when some scores "
        "equal 1. These exact breakpoints represent every attainable classification "
        "operating point; this is not an approximate evenly spaced search. Undefined "
        "metrics are retained as missing values. In particular, a rule predicting "
        "no positives has undefined precision and cannot satisfy the precision floor.\n\n"
        "The primary objective maximizes class-1 F1, then balanced accuracy, then recall, "
        "then proximity to 0.5, then the larger threshold. The secondary objective "
        f"maximizes recall among thresholds with precision >= {float(selection['precision_floor']):.2f}, "
        "then F1, then balanced accuracy, then the larger threshold. The precision floor "
        "was declared before threshold selection. Both objectives use only training OOF "
        "scores; no held-out labels or metrics enter the selection."
    )


def write_oof_report(output_dir, grid, selection):
    """Write the pre-freeze OOF analysis without accepting test metrics."""
    rows = _records(selection["oof_metrics"])
    if len(grid) != int(selection["candidate_count"]):
        raise ValueError("Threshold grid count disagrees with the frozen selection.")
    score_metrics = selection["oof_score_metrics"]
    text = f"""# Training-only threshold study

## Frozen neighbour model

Model selection already fixed **k=25, Euclidean distance, distance weighting**.
This study chooses a second-stage decision rule for that model. It does not rerun
the 64-configuration search or modify the accepted k=19/uniform baseline.

## Leakage-safe methodology

{_methodology(selection)}

## Selected OOF operating points

{_selection_description(selection)}

{_metric_tables(rows)}

{_changes(rows)}

## Precision/recall and confusion-count trade-off

{_tradeoffs(rows)}

Recall changes above are quantified in percentage points and detected defaults.
Whether they are practically material depends on the cost of missed defaults and
additional false alarms; no business costs were supplied. OOF performance helped
choose the thresholds and is not an unbiased estimate of their future performance.
The precision floor constrains training OOF selection and does not guarantee precision
on held-out or future data.

## Score-ranking metrics

For the single OOF score vector, **ROC-AUC = {float(score_metrics['roc_auc']):.6f}**
and **average precision = {float(score_metrics['average_precision']):.6f}**.
These ranking metrics are reported once: changing a decision threshold leaves
the scores and their ranking unchanged.

## Freeze and limitations

The thresholds and input hashes are frozen in `threshold_selection.json` before
held-out evaluation. This report contains training OOF results only. One fixed
training partition, five folds, and threshold selection on these OOF scores limit
generalization. Threshold tuning changes the operating point, not the neighbours
or weighting. Weka thresholds and illustrative cost-sensitive thresholds were
not evaluated in this study.
"""
    _write(Path(output_dir) / "threshold_oof_report.md", text)


def _save_figure(fig, stem):
    """Write reproducible PNG/SVG output without timestamp or random SVG IDs."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        with plt.rc_context({"svg.hashsalt": "knn-training-oof-threshold-study"}):
            fig.savefig(
                stem.with_suffix(".png"), dpi=160, bbox_inches="tight",
                metadata={"Software": "3804ICT threshold study"},
            )
            svg_path = stem.with_suffix(".svg")
            fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
            _write(svg_path, svg_path.read_text(encoding="utf-8"))
    finally:
        plt.close(fig)


def _mark_operating_points(ax, rows, x_metric, y_metric):
    for row in rows:
        name = row["threshold_name"]
        ax.scatter(
            float(row[x_metric]), float(row[y_metric]),
            label=f"{SHORT_LABELS[name]} (t={float(row['threshold']):.4f})",
            color=COLORS[name], marker=MARKERS[name], s=90 if name != "recall_oriented_threshold" else 150,
            edgecolors="black", linewidths=0.6, zorder=4,
        )


def plot_oof_figures(output_dir, grid, selection):
    """Create the three compact OOF views, marking only frozen operating points."""
    output_dir = Path(output_dir)
    rows = _records(selection["oof_metrics"])
    ordered = grid.sort_values("threshold", kind="stable")
    with plt.rc_context({"font.size": 9, "axes.titlesize": 11, "legend.fontsize": 8}):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(ordered.recall, ordered.precision, color="#8A8A8A", lw=1.0,
                label="OOF operating points", zorder=1)
        ax.axhline(float(selection["precision_floor"]), color="#54A24B", alpha=0.55,
                   linestyle=":", linewidth=1, label="Recall-objective precision floor")
        _mark_operating_points(ax, rows, "recall", "precision")
        ax.set(xlabel="Class-1 recall", ylabel="Class-1 precision",
               title="Training OOF precision/recall", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, output_dir / "threshold_precision_recall_curve")

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for metric, label, color in (
            ("f1", "F1", "#E45756"),
            ("recall", "Recall", "#54A24B"),
            ("precision", "Precision", "#4C78A8"),
            ("balanced_accuracy", "Balanced accuracy", "#B279A2"),
        ):
            ax.plot(ordered.threshold, ordered[metric], label=label, color=color, lw=1.25)
        for row in rows:
            ax.axvline(float(row["threshold"]), color=COLORS[row["threshold_name"]],
                       linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set(xlabel="Decision threshold (score >= threshold)", ylabel="OOF metric",
               title="Training OOF metrics by threshold", xlim=(0, 1), ylim=(-0.02, 1.02))
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, output_dir / "threshold_metrics_by_value")

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(ordered.FP, ordered.TP, color="#8A8A8A", lw=1.1,
                label="OOF operating points")
        _mark_operating_points(ax, rows, "FP", "TP")
        ax.set(xlabel="False positives (non-defaults flagged)",
               ylabel="True positives (defaults detected)", title="Training OOF confusion trade-off")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, output_dir / "threshold_confusion_tradeoff")


def _bootstrap_table(bootstrap):
    rows = []
    for row in _records(bootstrap):
        lower = float(row["bootstrap_95_percentile_lower"])
        upper = float(row["bootstrap_95_percentile_upper"])
        defined = np.isfinite(lower) and np.isfinite(upper)
        contains_zero = lower <= 0 <= upper
        rows.append([
            SHORT_LABELS[row["threshold_name"]], row["metric"],
            f"{float(row['observed_delta']):+.6f}",
            f"[{lower:+.6f}, {upper:+.6f}]" if defined else "Undefined",
            ("Includes zero" if contains_zero else "Excludes zero") if defined else "Unavailable",
            int(row["valid_resamples"]),
        ])
    return _table(
        ["Rule vs 0.5", "Metric", "Observed delta", "95% percentile interval",
         "Zero comparison", "Valid resamples"], rows,
    )


def _decomposition(rows, baseline):
    default = next(row for row in rows if row["threshold_name"] == "default_threshold")
    metrics = ("f1", "recall", "precision", "balanced_accuracy")
    changes = [[
        "Model configuration: k=19 uniform to k=25 distance at default rule",
        *[f"{float(default[name]) - float(baseline[name]):+.6f}" for name in metrics],
    ]]
    for row in rows:
        if row["threshold_name"] == "default_threshold":
            continue
        changes.append([
            f"Threshold only: 0.5 to {SHORT_LABELS[row['threshold_name']]}",
            *[f"{float(row[name]) - float(default[name]):+.6f}" for name in metrics],
        ])
    return _table(["Change", "F1 delta", "Recall delta", "Precision delta", "Balanced accuracy delta"], changes)


def _diagnostic_text(diagnostics):
    """Summarize agreement while retaining a link to all affected-row evidence."""
    if "threshold_agreement" in diagnostics:
        rows = []
        for name in THRESHOLD_NAMES:
            if name not in diagnostics["threshold_agreement"]:
                continue
            row = diagnostics["threshold_agreement"][name]
            rows.append([
                SHORT_LABELS[name], _threshold(row["threshold"]),
                int(row["label_disagreements"]),
                int(row["threshold_between_differing_scores"]),
                ", ".join(str(value) for value in row["row_ids"]) or "None",
            ])
        return (
            f"Across **{int(diagnostics['test_rows']):,}** test rows, "
            f"**{int(diagnostics['score_differences_above_1e_10'])}** score differences "
            "exceeded 1e-10. The maximum absolute score difference was "
            f"**{float(diagnostics['max_absolute_probability_difference']):.12g}**.\n\n"
            + _table(
                ["Rule", "Threshold", "Label disagreements", "Threshold separates scores",
                 "Disagreement row IDs"], rows,
            )
            + "\n\nThe `threshold_score_diagnostics.csv` file gives affected-row identifiers, "
            "both probabilities, causes, thresholded labels, and each threshold's "
            "distance to the interval between the two scores. The full agreement "
            "and provenance records are in `threshold_consistency.json`.\n\n"
            + f"Enhanced Custom repeatable: **{diagnostics.get('enhanced_repeatable', 'not recorded')}**. "
            + "Default-rule differences from the original selected-model labels: "
            + f"**{diagnostics.get('reference_default_label_differences', 'not recorded')}**; "
            + f"scores exactly equal to 0.5: **{diagnostics.get('exact_half_score_count', 'not recorded')}**."
        )

    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")

    return "```json\n" + json.dumps(diagnostics, indent=2, sort_keys=True, default=scalar) + "\n```"


def _conclusion(rows, baseline, bootstrap):
    default = next(row for row in rows if row["threshold_name"] == "default_threshold")
    model_f1_delta = float(default["f1"]) - float(baseline["f1"])
    model_recall_delta = float(default["recall"]) - float(baseline["recall"])
    paragraphs = [_tradeoffs(rows)]
    for row in rows:
        if row["threshold_name"] == "default_threshold":
            continue
        threshold_f1_delta = float(row["f1"]) - float(default["f1"])
        threshold_recall_delta = float(row["recall"]) - float(default["recall"])
        paragraphs.append(
            f"For {SHORT_LABELS[row['threshold_name']].lower()}, the threshold-only F1 "
            f"change ({threshold_f1_delta:+.6f}) was "
            f"{'larger than' if threshold_f1_delta > model_f1_delta else 'smaller than' if threshold_f1_delta < model_f1_delta else 'equal to'} "
            f"the earlier model-configuration change ({model_f1_delta:+.6f}); the "
            f"recall change ({threshold_recall_delta:+.6f}) was "
            f"{'larger than' if threshold_recall_delta > model_recall_delta else 'smaller than' if threshold_recall_delta < model_recall_delta else 'equal to'} "
            f"the model-configuration recall change ({model_recall_delta:+.6f}). "
            "These comparisons describe usefulness for F1 and default detection on this "
            "fixed split; they do not establish an overall business benefit."
        )
    for name in THRESHOLD_NAMES[1:]:
        intervals = [row for row in _records(bootstrap) if row["threshold_name"] == name
                     and row["metric"] in ("f1", "recall", "precision", "balanced_accuracy")]
        if not intervals:
            continue
        unavailable = [row["metric"] for row in intervals
                       if not np.isfinite(float(row["bootstrap_95_percentile_lower"]))
                       or not np.isfinite(float(row["bootstrap_95_percentile_upper"]))]
        intervals = [row for row in intervals if row["metric"] not in unavailable]
        includes = [row["metric"] for row in intervals
                    if float(row["bootstrap_95_percentile_lower"]) <= 0
                    <= float(row["bootstrap_95_percentile_upper"])]
        excludes = [row["metric"] for row in intervals if row["metric"] not in includes]
        pieces = []
        if includes:
            pieces.append("include zero for " + ", ".join(includes))
        if excludes:
            pieces.append("exclude zero for " + ", ".join(excludes))
        if unavailable:
            pieces.append("are unavailable for " + ", ".join(unavailable))
        paragraphs.append(
            f"The {SHORT_LABELS[name].lower()} paired bootstrap intervals "
            + " and ".join(pieces)
            + ". Intervals containing zero leave the direction of those changes uncertain "
            "under this resampling. Intervals excluding zero support a consistent direction "
            "within the observed test sample, not robustness to new populations."
        )
    paragraphs.append(
        "Recall gains are quantified above; their practical materiality depends on the "
        "extra false positives and unspecified business costs. Threshold tuning changes "
        "the precision/recall operating point and leaves ROC-AUC and average precision "
        "unchanged. The evidence does not make the model universally better."
    )
    return "\n\n".join(paragraphs)


def write_test_report(output_dir, comparison, bootstrap, selection, baseline, score_metrics, diagnostics):
    """Write the final interpretation after evaluation of already-frozen thresholds."""
    rows = _records(comparison)
    bootstrap_rows = _records(bootstrap)
    bootstrap_settings = sorted({(int(row["resamples"]), int(row["seed"])) for row in bootstrap_rows})
    settings_text = "; ".join(f"{count:,} resamples with seed {seed}" for count, seed in bootstrap_settings)
    text = f"""# Frozen-model decision-threshold study

## 1. Objective

Assess whether a decision threshold selected only from training OOF scores improves
class-1/default F1 and recall for the already-selected KNN. All changes below are
observed on the fixed split; thresholds were not chosen from test performance.

## 2. Frozen model

The neighbour model is **k=25, Euclidean distance, distance weighting**. The accepted
V5.1 reference remains **k=19, Euclidean, uniform, default argmax**. Neither model
selection nor the accepted implementation was modified for this threshold study.

## 3. Leakage-safe OOF methodology

{_methodology(selection)}

The threshold selection and relevant input hashes were frozen before final
evaluation. The OOF report and figures were generated before the test evaluation.
The test partition was excluded from second-stage threshold selection; its
threshold-0.5 result had already been reported by the earlier model-quality study.

## 4. OOF threshold results

{_metric_tables(_records(selection['oof_metrics']))}

The single OOF score vector had ROC-AUC **{float(selection['oof_score_metrics']['roc_auc']):.6f}**
and average precision **{float(selection['oof_score_metrics']['average_precision']):.6f}**.
These ranking metrics do not change with the threshold.

## 5. Frozen thresholds

{_selection_description(selection)}

The OOF precision floor is a selection constraint, not a guarantee of test or future
precision. Ordinary commands refuse to overwrite a completed study. An explicit
`--verify-rerun` checks frozen hashes and thresholds and repeats evaluation without
selecting a new threshold.

## 6. Untouched-test results

Preprocessing was fitted once to all 24,000 training rows. The primary score vector
comes from EnhancedCustomKNNClassifier with the frozen model configuration. The
three decision rules use that same vector and the thresholds fixed above.

{_metric_tables(rows, baseline)}

For the one Enhanced Custom test score vector, **ROC-AUC = {float(score_metrics['roc_auc']):.6f}**
and **average precision = {float(score_metrics['average_precision']):.6f}**. Threshold
tuning changes neither score-ranking metric. The previously reported default
threshold result is retained as the reference for all threshold-only deltas.

## 7. Model-change vs threshold-change decomposition

{_decomposition(rows, baseline)}

The first row compares different neighbour configurations at their default rules.
The remaining rows change only the selected model's operating point. These sources
of change are reported separately.

## 8. Precision/recall trade-off

{_tradeoffs(rows)}

An OOF-selected precision floor need not hold on the test set. A recall gain can
come with lower precision and lower accuracy; this is not a universal improvement.

## 9. Confusion-matrix changes

{_changes(rows)}

All differences are candidate minus selected-model threshold 0.5, except FN avoided,
which is reference FN minus candidate FN. Positive extra TP counts additional
defaults detected; positive extra FP counts additional non-defaults flagged.

## 10. Bootstrap intervals

The paired bootstrap resampled the same test-row indices for both decision rules:
{settings_text}. Each threshold candidate is compared with the selected model at
0.5. The table reports 95% percentile intervals and explicitly states whether each
interval includes zero. Count deltas refer to resamples of the original test size.

{_bootstrap_table(bootstrap)}

These are conditional resampling intervals for this fixed split and frozen model.
They are not automatically labelled statistically significant and do not account
for training-data variation, model selection, threshold selection, or distribution
shift. No threshold was adjusted in response to these intervals.

## 11. Probability-semantics caveat

Enhanced Custom is the primary implementation; scikit-learn scores provide a
diagnostic comparison. The earlier model-quality evaluation identified seven
probability-difference rows associated with exact-zero handling, boundary ties,
and floating-point distance arithmetic. The new diagnostic evidence below checks
the default and both selected threshold rules, including whether a selected
threshold falls between the differing scores. Row identifiers and probabilities
for any label disagreements are retained in the study diagnostics.

{_diagnostic_text(diagnostics)}

Weka numeric thresholds were **not evaluated**. Genuine Weka IBk has native
inverse-distance and nominal-probability semantics, so the same numeric threshold
must not be assumed equivalent. Applying these thresholds to Weka would require
a separately labelled exploratory appendix.

## 12. Optional cost-sensitive appendix

Not run. Business costs were not supplied. An illustrative training-OOF analysis
of FP + ratio * FN at FN/FP ratios 1, 2, 5 and 10 could be studied separately; it
would not replace the primary F1 objective or justify changing a frozen threshold
after test inspection.

## 13. Limitations

Only one fixed dataset split and five OOF folds were used. The best OOF thresholds
were selected using the same OOF scores summarized here, so OOF gains can be
optimistic. The test partition had previously been evaluated at the default rule,
but no test scores, labels or metrics were used to choose these thresholds. A
single held-out sample and a paired bootstrap do not establish external validity.
The OOF precision floor is not a population guarantee. Probability estimates can
differ slightly between exact implementations near zero distances and boundary
ties. Neither probability calibration nor business costs were established.

## 14. Conclusion

{_conclusion(rows, baseline, bootstrap)}
"""
    _write(Path(output_dir) / "threshold_report.md", text)
