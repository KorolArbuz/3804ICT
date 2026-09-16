"""Regenerate final tables and figures from saved scores and timings, without inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.common.experiment import IMPLEMENTATIONS, load_configs, markdown_table, model_group, sha256
from src.evaluation.final import metrics as calculate_metrics, read_predictions

LABELS = dict(zip(IMPLEMENTATIONS, ("Python V5.1 baseline", "Custom Python", "scikit-learn", "Custom C++20", "Java/Weka IBk")))
COLORS = dict(zip(IMPLEMENTATIONS, ("#777777", "#0072B2", "#E69F00", "#009E73", "#CC79A7")))
FINALS = IMPLEMENTATIONS[1:]


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "3804ICT-final-figures"
    import matplotlib.pyplot as plt
    return plt


def save(figure, output):
    figure.tight_layout()
    paths = []
    for suffix in (".png", ".svg"):
        path = output.with_suffix(suffix)
        metadata = {"Date": None} if suffix == ".svg" else {"Software": "3804ICT saved-result report"}
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", metadata=metadata)
        paths.append(path.name)
    pyplot().close(figure)
    return paths


def load_evidence(run_dir):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    configurations = load_configs()
    expected_hashes = {name: config["config_hash"] for name, config in configurations.items()}
    if manifest.get("config_hashes") != expected_hashes:
        raise ValueError("Saved run configuration hashes differ from frozen configurations")
    table = pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")
    if table.empty or table.implementation_id.duplicated().any() or not set(table.implementation_id).issubset(IMPLEMENTATIONS):
        raise ValueError("Duplicate or unknown implementation in saved metrics")
    frames, identities = {}, None
    inputs = [run_dir / "metrics.csv", run_dir / "agreement.csv"]
    for implementation in table.implementation_id:
        path = run_dir / "predictions" / f"{implementation}.csv"
        frame = read_predictions(path)
        config = configurations[model_group(implementation)]
        if (frame.empty or not (frame.config_hash == config["config_hash"]).all()
                or not (frame.run_id == manifest["run_id"]).all()
                or not (frame.implementation_id == implementation).all()):
            raise ValueError(f"Stale prediction identity/configuration: {implementation}")
        scores = frame.score_class_1.to_numpy(dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid saved scores")
        if not np.isin(frame.y_true, [0, 1]).all() or not np.isin(frame.y_pred, [0, 1]).all():
            raise ValueError("Nonbinary saved labels")
        if frame.row_id.duplicated().any() or not np.array_equal(frame.test_position, np.arange(len(frame))):
            raise ValueError("Unaligned saved prediction rows")
        current = frame[["row_id", "original_id", "y_true"]]
        if identities is not None and not identities.equals(current):
            raise ValueError("Implementations have different row identities or ground truth")
        identities = current
        threshold = config["model"]["threshold"]
        expected = scores > .5 if threshold is None else scores >= threshold
        if not np.array_equal(expected.astype(int), frame.y_pred.to_numpy()):
            raise ValueError("Saved labels disagree with the frozen decision rule")
        calculated = calculate_metrics(frame)
        saved = table.loc[table.implementation_id == implementation].iloc[0]
        for key, value in calculated.items():
            matches = saved[key] == value if isinstance(value, str) else np.isclose(saved[key], value, rtol=0, atol=1e-12)
            if not matches:
                raise ValueError(f"Saved metric arithmetic differs: {implementation}/{key}")
        frames[implementation] = frame
        inputs.append(path)
    agreement = pd.read_csv(run_dir / "agreement.csv", float_precision="round_trip")
    for row in agreement.itertuples(index=False):
        if row.implementation_a not in frames or row.implementation_b not in frames:
            raise ValueError("Agreement references an absent implementation")
        a, b = frames[row.implementation_a], frames[row.implementation_b]
        disagreements = int(np.count_nonzero(a.y_pred.to_numpy() != b.y_pred.to_numpy()))
        if disagreements != row.label_disagreements:
            raise ValueError("Saved disagreement count differs from predictions")
        delta = np.abs(a.score_class_1.to_numpy() - b.score_class_1.to_numpy())
        if (row.rows != len(a) or row.scores_over_tolerance != np.count_nonzero(delta > row.score_tolerance)
                or not np.isclose(row.label_agreement, 1 - disagreements / len(a), rtol=0, atol=1e-12)
                or not np.isclose(row.max_abs_score_difference, delta.max(), rtol=0, atol=1e-12)
                or not np.isclose(row.mean_abs_score_difference, delta.mean(), rtol=0, atol=1e-12)):
            raise ValueError("Saved score comparison differs from predictions")
    return manifest, table, agreement, frames, inputs


def plot_quality(table, output):
    figure, axis = pyplot().subplots(figsize=(10, 5))
    columns = ["f1", "recall", "precision", "average_precision", "roc_auc"]
    positions, width = np.arange(len(columns)), .8 / max(len(table), 1)
    for index, row in enumerate(table.itertuples(index=False)):
        axis.bar(positions + (index - (len(table) - 1) / 2) * width,
                 [getattr(row, name) for name in columns], width,
                 label=LABELS[row.implementation_id], color=COLORS[row.implementation_id])
    axis.set(xticks=positions, xticklabels=["F1", "Recall", "Precision", "Average precision", "ROC-AUC"],
             ylim=(0, 1), ylabel="Score on fixed legacy test rows",
             title="Quality: baseline model and frozen final implementations")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=9)
    axis.grid(axis="y", alpha=.2)
    return save(figure, output / "quality_comparison")


def plot_disagreements(agreement, present, output):
    names = [name for name in FINALS if name in present]
    matrix = np.full((len(names), len(names)), np.nan)
    np.fill_diagonal(matrix, 0)
    for row in agreement.itertuples(index=False):
        if row.implementation_a in names and row.implementation_b in names:
            i, j = names.index(row.implementation_a), names.index(row.implementation_b)
            matrix[i, j] = matrix[j, i] = row.label_disagreements
    figure, axis = pyplot().subplots(figsize=(7, 5.5))
    rendered = axis.imshow(matrix, cmap="Blues", vmin=0)
    for (i, j), count in np.ndenumerate(matrix):
        color = "white" if np.isfinite(count) and count > np.nanmax(matrix) / 2 else "black"
        axis.text(j, i, "unavailable" if np.isnan(count) else str(int(count)), ha="center", va="center", color=color)
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=[LABELS[name] for name in names],
             yticklabels=[LABELS[name] for name in names], title="Final implementations: pairwise label disagreements")
    axis.tick_params(axis="x", rotation=20)
    figure.colorbar(rendered, ax=axis, label="Rows with different labels")
    return save(figure, output / "pairwise_disagreements")


def plot_confusion(table, output):
    row = table.loc[table.implementation_id == "final_python"].iloc[0]
    matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]], dtype=int)
    figure, axis = pyplot().subplots(figsize=(5.5, 5))
    rendered = axis.imshow(matrix, cmap="Blues", vmin=0)
    for (i, j), count in np.ndenumerate(matrix):
        axis.text(j, i, str(count), ha="center", va="center", fontsize=16,
                  color="white" if count > matrix.max() / 2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"], yticklabels=["No default", "Default"],
             xlabel="Predicted", ylabel="Actual", title="Final Custom Python: balanced_low_fp")
    figure.colorbar(rendered, ax=axis, label="Rows")
    return save(figure, output / "confusion_matrix_final_python")


def plot_precision_recall(frames, output):
    figure, axis = pyplot().subplots(figsize=(8, 5))
    # Final toolkit curves almost overlap; the tables retain every implementation.
    for name in ("baseline_python_v5_1", "final_python"):
        if name in frames:
            frame = frames[name]
            precision, recall, _ = precision_recall_curve(frame.y_true, frame.score_class_1)
            axis.plot(recall, precision, label=LABELS[name], color=COLORS[name])
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
             title="Saved scores on the previously inspected legacy test")
    axis.legend()
    axis.grid(alpha=.2)
    return save(figure, output / "precision_recall")


def prediction_timings(run_dir):
    path = run_dir / "benchmark_raw.csv"
    if not path.is_file():
        return None
    raw = pd.read_csv(path, float_precision="round_trip")
    if raw.empty:
        return None
    required = {"run_id", "config_hash", "implementation_id", "phase", "seconds", "trial", "sequence", "is_warmup"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Benchmark CSV missing columns: {sorted(required - set(raw.columns))}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    configs = load_configs()
    for row in raw.itertuples(index=False):
        if row.run_id != manifest["run_id"] or row.config_hash != configs[model_group(row.implementation_id)]["config_hash"]:
            raise ValueError("Benchmark run identity or configuration hash differs")
    selected = raw[raw.phase.isin({"prediction", "predict", "prediction_only", "prediction_seconds"})].copy()
    selected = selected[~selected.is_warmup.astype(str).str.lower().isin({"true", "1"})]
    if selected.empty:
        raise ValueError("Benchmark has no measured prediction-phase samples")
    if not np.isfinite(selected.seconds).all() or (selected.seconds < 0).any():
        raise ValueError("Invalid measured durations")
    return selected


def plot_runtime(timings, output):
    names = [name for name in FINALS if (timings.implementation_id == name).any()]
    if not names:
        return []
    values = [timings.loc[timings.implementation_id == name, "seconds"].to_numpy() for name in names]
    quantiles = np.array([np.quantile(value, [.25, .5, .75]) for value in values])
    figure, axis = pyplot().subplots(figsize=(8, 5))
    axis.bar(np.arange(len(names)), quantiles[:, 1], color=[COLORS[name] for name in names],
             yerr=[quantiles[:, 1] - quantiles[:, 0], quantiles[:, 2] - quantiles[:, 1]], capsize=6)
    axis.set(xticks=np.arange(len(names)), xticklabels=[LABELS[name] for name in names], ylabel="Seconds",
             title="Final workload: prediction median and interquartile range")
    axis.grid(axis="y", alpha=.2)
    paths = save(figure, output / "final_prediction_runtime")
    figure, axis = pyplot().subplots(figsize=(8, 5))
    boxes = axis.boxplot(values, labels=[LABELS[name] for name in names], patch_artist=True, showfliers=True)
    for box, name in zip(boxes["boxes"], names):
        box.set_facecolor(COLORS[name])
        box.set_alpha(.6)
    axis.set(ylabel="Seconds", title="Final workload: prediction-time distribution")
    axis.grid(axis="y", alpha=.2)
    paths += save(figure, output / "runtime_distribution")
    figure, axis = pyplot().subplots(figsize=(9, 5))
    for name in names:
        rows = timings[timings.implementation_id == name].sort_values("sequence")
        axis.plot(rows.sequence, rows.seconds, "o-", label=LABELS[name], color=COLORS[name])
    axis.set(xlabel="Measured process sequence (warmups omitted)", ylabel="Seconds", title="Final workload: run-order stability")
    axis.legend()
    axis.grid(alpha=.2)
    return paths + save(figure, output / "run_order_stability")


def plot_results(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    output = Path(output_dir) if output_dir else run_dir / "figures"
    manifest, table, agreement, frames, inputs = load_evidence(run_dir)
    timings, omissions = prediction_timings(run_dir), []
    output.mkdir(parents=True, exist_ok=True)
    paths = plot_quality(table, output)
    if any(name in frames for name in FINALS):
        paths += plot_disagreements(agreement, frames, output)
    if "final_python" in frames:
        paths += plot_confusion(table, output)
    paths += plot_precision_recall(frames, output)
    if timings is not None:
        paths += plot_runtime(timings, output)
        inputs.append(run_dir / "benchmark_raw.csv")
    else:
        omissions.append("Runtime figures omitted: no benchmark_raw.csv for this run. Integration durations are not substituted for repeated benchmark measurements.")
    columns = ["implementation_id", "f1", "recall", "precision", "average_precision", "roc_auc", "accuracy", "balanced_accuracy", "TN", "FP", "FN", "TP"]
    shown = table[columns].copy()
    for column in columns[1:8]:
        shown[column] = shown[column].map(lambda value: f"{value:.6f}")
    text = ["# Final comparison", "", f"Run: `{manifest['run_id']}`; manifest: [run_manifest.json](run_manifest.json).", "",
            "These are fixed-data software reproduction results on a previously inspected legacy test. They are not new independent statistical validation.", "",
            "The V5.1 baseline uses a different model and workload (k=19, uniform vote, original features). The four final implementations use k=101, inverse-distance voting and the frozen balanced_low_fp threshold. Weka retains native IBk distribution semantics.", "",
            markdown_table(shown), "", "## Final implementation agreement", "", markdown_table(agreement), "",
            "The training-OOF recall >= 0.50 constraint is a selection criterion; it does not guarantee test recall.", ""]
    if timings is not None:
        summary = timings.groupby("implementation_id").seconds.agg(n="count", median_seconds="median", min_seconds="min", max_seconds="max")
        summary["q25_seconds"] = timings.groupby("implementation_id").seconds.quantile(.25)
        summary["q75_seconds"] = timings.groupby("implementation_id").seconds.quantile(.75)
        summary["iqr_seconds"] = summary.q75_seconds - summary.q25_seconds
        text += ["## Runtime on the common final workload", "", "Prediction phase only; warmups excluded. Fit, startup, loading and end-to-end scopes remain separately recorded in benchmark CSVs.", "",
                 markdown_table(summary.loc[summary.index.isin(FINALS)].reset_index()), ""]
        if "baseline_python_v5_1" in summary.index:
            text += ["## Baseline timing: different model and workload", "", markdown_table(summary.loc[["baseline_python_v5_1"]].reset_index()), "",
                     "A baseline/final timing ratio is not a same-model implementation speedup.", ""]
    text += omissions
    (run_dir / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    report = {"run_id": manifest["run_id"], "config_hashes": manifest["config_hashes"],
              "source": "Saved results in this run only; no raw-data loading, fitting, inference, JVM or compiler",
              "inputs": {path.relative_to(run_dir).as_posix(): sha256(path) for path in inputs},
              "figures": paths, "omissions": omissions}
    (run_dir / "report_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


generate_report = plot_results
