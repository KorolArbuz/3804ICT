"""Focused figures and the fixed-structure Model Quality V2 report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import STAGE_NAMES


DISCLAIMER = (
    "Legacy held-out test comparison; test was already examined in earlier project "
    "stages and is not independent validation for V2."
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(fig, output_dir, name):
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = output_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            clean = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
            path.write_text(clean + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def _stage_metric_plot(outer, metric, output_dir, name, label):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    stages = sorted(outer.stage.unique())
    for role, marker in (("baseline", "o"), ("candidate", "s")):
        means, spreads = [], []
        for stage in stages:
            values = outer[(outer.stage == stage) & (outer.role == role)][metric]
            means.append(values.mean())
            spreads.append(values.std(ddof=1))
        ax.errorbar(stages, means, yerr=spreads, marker=marker, capsize=4,
                    linewidth=1.7, label=role.capitalize())
    ax.set_xticks(stages, [f"S{stage}" for stage in stages])
    ax.set_xlabel("Staged hypothesis")
    ax.set_ylabel(label)
    ax.set_title(f"Nested outer-fold {label}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, output_dir, name)


def create_figures(output_dir, outer, inner, decisions):
    _stage_metric_plot(outer, "average_precision", output_dir, "nested_stage_ap",
                       "Average Precision")
    _stage_metric_plot(outer, "roc_auc", output_dir, "nested_stage_auc", "ROC-AUC")
    _stage_metric_plot(outer, "f1", output_dir, "nested_stage_f1", "threshold-selected F1")

    candidate = outer[outer.role == "candidate"]
    means = candidate.groupby("stage")[["recall", "precision"]].mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(means.index, means.recall, marker="o", label="Recall")
    ax.plot(means.index, means.precision, marker="s", label="Precision")
    ax.set_xticks(means.index, [f"S{stage}" for stage in means.index])
    ax.set_xlabel("Staged hypothesis")
    ax.set_ylabel("Outer-fold mean")
    ax.set_title("Nested threshold operating points")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, output_dir, "nested_stage_recall_precision")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = [f"S{item['stage']}\n{'accept' if item['accepted'] else 'retain'}" for item in decisions]
    ap = [item["paired_deltas"]["average_precision"]["mean_delta"] for item in decisions]
    auc = [item["paired_deltas"]["roc_auc"]["mean_delta"] for item in decisions]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, ap, width=0.36, label="AP delta")
    ax.bar(x + 0.18, auc, width=0.36, label="ROC-AUC delta")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Candidate minus baseline, outer-fold mean")
    ax.set_title("Staged ablation decisions")
    ax.legend()
    _save(fig, output_dir, "stage_ablation_summary")

    stage4 = inner[(inner.stage == 4) & (inner.outer_fold == 0)
                   & (inner.row_type == "full_training_summary")].copy()
    if not stage4.empty:
        stage4 = stage4.sort_values("configuration_id")
        fig, ax = plt.subplots(figsize=(10, 5.2))
        positions = np.arange(len(stage4))
        ax.bar(positions - 0.18, stage4.average_precision, width=0.36, label="AP")
        ax.bar(positions + 0.18, stage4.roc_auc, width=0.36, label="ROC-AUC")
        blocks = stage4.configuration_id.str.extract(r"blocks-([^_]+)")[0]
        ax.set_xticks(positions, blocks)
        ax.set_ylabel("Three-fold full-training OOF score")
        ax.set_title("Predeclared engineered feature blocks")
        ax.legend()
        _save(fig, output_dir, "feature_block_comparison")


def _table(frame, columns):
    return frame[list(columns)].to_markdown(index=False, floatfmt=".6f")


def create_reports(output_dir):
    output_dir = Path(output_dir)
    outer = pd.read_csv(output_dir / "nested_outer_results.csv")
    inner = pd.read_csv(output_dir / "nested_inner_selection.csv")
    decisions = _read_json(output_dir / "stage_decisions.json")
    final = _read_json(output_dir / "final_v2_configuration.json")
    manifest = _read_json(output_dir / "reference_manifest.json")
    parity = _read_json(output_dir / "manual_v2_parity.json")
    legacy = pd.read_csv(output_dir / "legacy_test_comparison.csv")
    create_figures(output_dir, outer, inner, decisions)

    candidate = outer[outer.role == "candidate"]
    stage_rows = []
    for decision in decisions:
        stage = decision["stage"]
        values = candidate[candidate.stage == stage]
        stage_rows.append({
            "stage": stage, "hypothesis": STAGE_NAMES[stage],
            "decision": "accepted" if decision["accepted"] else "retained baseline",
            "AP": values.average_precision.mean(), "ROC-AUC": values.roc_auc.mean(),
            "F1": values.f1.mean(), "AP wins": decision["paired_deltas"]["average_precision"]["wins"],
        })
    stage_frame = pd.DataFrame(stage_rows)
    final_role = "candidate" if decisions[-1]["accepted"] else "baseline"
    final_outer = outer[(outer.stage == 5) & (outer.role == final_role)].sort_values("outer_fold")
    first_baseline = outer[(outer.stage == 1) & (outer.role == "baseline")]
    observed_ap = final_outer.average_precision.mean() - first_baseline.average_precision.mean()
    observed_auc = final_outer.roc_auc.mean() - first_baseline.roc_auc.mean()
    improved = observed_ap > 0 and observed_auc >= -0.001
    conclusion = (
        "The staged final procedure showed higher outer-fold score ranking than the starting "
        "k=25 representation" if improved else
        "The staged evidence did not show a reliable score-ranking improvement over the starting "
        "k=25 representation"
    )
    config = final["configuration"]
    report = f"""# Model Quality V2

## 1. Research question

Can KNN improve underlying score ranking and neighbour quality through feature representation, k selection, and neighbourhood geometry? Average Precision (AP) is the primary metric, followed by ROC-AUC. Threshold-selected classification metrics describe the operating point and are not evidence that ranking itself improved.

## 2. Why nested CV

All selection used five outer folds and three inner folds inside the fixed 24,000-row training partition (shuffle seed 42). Inner OOF predictions selected both configuration and threshold. Each transform was fitted on its inner-training rows. The outer labels were used once for evaluation and never entered configuration or threshold selection. The same outer folds were reused across the staged investigation, which makes the stage comparisons paired but also adaptive.

## 3. Current reference model

The starting model is k=25, Euclidean distance, inverse-distance voting, and the existing preprocessing. The earlier threshold study selected {manifest['threshold_study']['f1_optimized_threshold']:.9f}; it improved the classification operating point, while its AP and ROC-AUC remained score-ranking properties of the unchanged model.

## 4. Stage 1 k+threshold

The candidates were k=13, 19, 25, 31, 35, 51, 75, and 101. Each was ranked by mean inner-fold AP, then mean ROC-AUC, OOF threshold-selected F1, balanced accuracy, and smaller k. The retained configuration after this stage was `{decisions[0]['retained_configuration_id']}`; the stage was **{'accepted' if decisions[0]['accepted'] else 'rejected'}**.

## 5. Stage 2 money transforms

Standard scaling, signed-log plus standard scaling, Yeo-Johnson, and robust scaling were compared only for limit, bill, and payment amounts. The retained configuration was `{decisions[1]['retained_configuration_id']}`; the stage was **{'accepted' if decisions[1]['accepted'] else 'rejected'}**.

## 6. Stage 3 PAY representation

Raw scaled PAY codes were compared with fold-learned positive-delay values and neutral indicators for observed non-positive codes. Unseen categories produce zero for every learned category indicator. The retained configuration was `{decisions[2]['retained_configuration_id']}`; the stage was **{'accepted' if decisions[2]['accepted'] else 'rejected'}**.

## 7. Stage 4 engineered features

The ablation compared no block, limit-relative Block A, delinquency-summary Block B, and A+B. Ratios use missing values for missing or non-positive denominators; fold-fitted median imputation handles them. These are descriptive features, not causal variables. The retained configuration was `{decisions[3]['retained_configuration_id']}`; the stage was **{'accepted' if decisions[3]['accepted'] else 'rejected'}**.

## 8. Stage 5 group weighting

PAY/delinquency squared-distance weights 0.5, 1.0, and 2.0 were implemented by multiplying standardized group columns by the square root of the weight. The retained configuration was `{decisions[4]['retained_configuration_id']}`; the stage was **{'accepted' if decisions[4]['accepted'] else 'rejected'}**.

{_table(stage_frame, ['stage', 'hypothesis', 'decision', 'AP', 'ROC-AUC', 'F1', 'AP wins'])}

## 9. Final configuration

The configuration was frozen before any V2 legacy-test scoring: k={config['k']}, Euclidean distance, distance voting, `{config['money_transform']}` money transform, `{config['pay_representation']}` PAY representation, feature blocks `{config['feature_blocks']}`, PAY/delinquency weight {config['pay_group_weight']}, and nested-derived threshold {final['threshold']:.9f}. The acceptance rule required AP improvement of at least 0.002 or AP wins in at least four folds, with mean ROC-AUC decrease no worse than 0.001 and F1 decrease no worse than 0.002.

## 10. Manual V2 parity

The clarity-first manual kernel uses exact brute-force Euclidean neighbours, inverse-distance voting, exact zero-distance semantics, and original-index boundary tie handling. Against scikit-learn on {parity['rows']} transformed legacy and selected outer-fold example rows, the maximum absolute probability difference was {parity['maximum_absolute_probability_difference']:.3g} (tolerance {parity['tolerance']:.1g}); probability and prediction parity **{'passed' if parity['passed'] and parity['passed_predictions'] else 'failed'}**.

## 11. Nested-CV evidence

Final-stage outer-fold values:

{_table(final_outer, ['outer_fold', 'average_precision', 'roc_auc', 'f1', 'recall', 'precision', 'balanced_accuracy', 'threshold'])}

Relative to the Stage-1 starting baseline, the final-stage adaptive procedure changed mean AP by {observed_ap:+.6f} and mean ROC-AUC by {observed_auc:+.6f}. Raw paired deltas and fold win counts for every stage are in `stage_decisions.csv`.

## 12. Legacy test comparison

**{DISCLAIMER}** No selection was reopened after these values were inspected.

{_table(legacy, ['comparison', 'average_precision', 'roc_auc', 'f1', 'recall', 'precision', 'balanced_accuracy', 'TN', 'FP', 'FN', 'TP'])}

## 13. Limitations

Five outer folds give only five paired observations, so the deltas are descriptive engineering evidence rather than significance claims. Stages reuse the same folds and later hypotheses depend on earlier decisions. Only the predeclared, small candidate families were tested. The legacy test has prior analyst exposure. PAY code indicators use neutral labels because undocumented code meanings were not assumed.

## 14. Conclusion

{conclusion}. The observed mean changes were AP {observed_ap:+.6f} and ROC-AUC {observed_auc:+.6f}. Threshold selection separately changed the class-1 operating point; it did not itself improve AP or ROC-AUC.
"""
    (output_dir / "final_v2_report.md").write_text(report, encoding="utf-8", newline="\n")
