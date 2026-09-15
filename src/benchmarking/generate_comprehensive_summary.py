"""Write the comprehensive academic benchmark narrative from generated tables."""

import shutil
from collections import defaultdict
from pathlib import Path

from .benchmark_utils import BENCHMARK_DIR, DISPLAY_NAMES, IMPLEMENTATIONS, read_json, read_rows, write_json


def _fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def _runtime_table(rows):
    lines = [
        "| Implementation | Runs | Median (s) | Min | Max | IQR | CV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {DISPLAY_NAMES[row['implementation']]} | {row['runs']} | "
            f"{_fmt(row['median_seconds'])} | {_fmt(row['min_seconds'])} | "
            f"{_fmt(row['max_seconds'])} | {_fmt(row['iqr_seconds'])} | "
            f"{100 * float(row['coefficient_of_variation']):.2f}% |"
        )
    return "\n".join(lines)


def _iqr_outliers():
    rows = [row for row in read_rows(BENCHMARK_DIR / "raw_prediction_runs.csv") if row["warmup"] == "false"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["implementation"]].append(row)
    result = {}
    for name, values in grouped.items():
        ordered = sorted(float(row["runtime_seconds"]) for row in values)
        q1 = (ordered[4] + ordered[5]) / 2
        q3 = (ordered[14] + ordered[15]) / 2
        iqr = q3 - q1
        result[name] = [
            {
                "global_run_order": int(row["global_run_order"]),
                "run_number": int(row["run_number_within_implementation"]),
                "runtime_seconds": float(row["runtime_seconds"]),
            }
            for row in values
            if float(row["runtime_seconds"]) < q1 - 1.5 * iqr
            or float(row["runtime_seconds"]) > q3 + 1.5 * iqr
        ]
    return result


def _history_table(rows):
    lines = [
        "| Version | Status | Runtime (s) | Speedup vs Original | Main change |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        status = "Historical/accepted path" if row["accepted"] == "true" else "Rejected" if row["version"] in {"V6", "V7"} else "Experimental"
        lines.append(
            f"| {row['version']} | {status} | {_fmt(row['median_runtime_seconds'])} | "
            f"{float(row['speedup_vs_original']):.1f}x | {row['main_change']} |"
        )
    return "\n".join(lines)


def _quality_table(rows):
    labels = {"custom": "Python V5.1", "sklearn": "scikit-learn", "weka": "Weka IBk", "custom_cpp": "C++20 experimental"}
    lines = [
        "| Implementation | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['implementation']]} | {float(row['accuracy']):.4f} | "
            f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['specificity']):.4f} | {float(row['f1']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['roc_auc']):.4f} | "
            f"{float(row['average_precision']):.4f} |"
        )
    return "\n".join(lines)


def main():
    summary = read_json(BENCHMARK_DIR / "runtime_summary.json")
    environment = summary["environment"]
    prediction = {row["implementation"]: row for row in summary["prediction_only"]}
    pipeline = {row["implementation"]: row for row in summary["full_pipeline"]}
    fit = {row["implementation"]: row for row in summary["fit_build"]}
    outliers = _iqr_outliers()
    total_outliers = sum(len(items) for items in outliers.values())
    cpp_outlier = outliers["custom_cpp"][0] if outliers["custom_cpp"] else None
    throughput = {name: 6000 / prediction[name]["median_seconds"] for name in IMPLEMENTATIONS}
    milliseconds = {name: 1000 * prediction[name]["median_seconds"] / 6000 for name in IMPLEMENTATIONS}
    train_regression = summary["training_size_regression"]
    query_regression = summary["query_size_regression"]
    selected_cpp = next(row for row in summary["cpp_configurations"] if row["selected_main_configuration"] == "true")
    portable_cpp = next(
        row for row in summary["cpp_configurations"]
        if row["build"] == "portable" and row["selection_method"] == "bounded max-heap" and row["batch_size"] == 32
    )
    custom_workspace = next(row for row in summary["memory"] if row["implementation"] == "custom_python_v5_1" and row["category"] == "major_prediction_workspace")
    cpp_workspace = next(row for row in summary["memory"] if row["implementation"] == "custom_cpp" and row["category"] == "major_prediction_workspace")
    history_v51 = next(row for row in summary["historical_optimization"] if row["version"] == "V5.1")

    initial_md = BENCHMARK_DIR / "benchmark_summary_prediction_pipeline.md"
    initial_json = BENCHMARK_DIR / "benchmark_summary_prediction_pipeline.json"
    if not initial_md.exists() and (BENCHMARK_DIR / "benchmark_summary.md").exists():
        shutil.copyfile(BENCHMARK_DIR / "benchmark_summary.md", initial_md)
    if not initial_json.exists() and (BENCHMARK_DIR / "benchmark_summary.json").exists():
        shutil.copyfile(BENCHMARK_DIR / "benchmark_summary.json", initial_json)

    markdown = f"""# Comprehensive KNN benchmark and runtime analysis

## 1. Purpose

This suite measures runtime, stability, scaling, memory, correctness, classification quality, and compiler/configuration effects without changing the accepted Python V5.1, scikit-learn, or Weka model semantics. The pure C++20 implementation is evaluated as an experimental exact candidate.

**Measured fact.** All new raw observations are retained. Primary statistics include every timed run; no outlier was removed or smoothed.

## 2. Experimental setup

The controlled prediction benchmark used 20 randomized, interleaved timed runs per implementation after one recorded warm-up. Every run used the same 24,000 by 33 training matrix, 6,000-row test matrix, `k=19`, Euclidean distance, uniform voting, row order, and one-thread policy. Python measured `predict_proba` plus class selection. C++ used its internal timer after loading and an internal warm-up. Weka used the existing internal `distributionForInstance` loop timer, excluding JVM startup, loading, build, and output.

The full prepared-input pipeline used five randomized runs per implementation. It includes processed input loading, fit/build, prediction, required output writing, and metric calculation. Shared raw-data preprocessing, train/test splitting, cross-validation, and `k` selection are fixed setup and excluded.

Training-size scaling used nested prefixes from 1,000 to 24,000 training rows with the first 1,000 test queries fixed. Query scaling used all 24,000 training rows and nested prefixes from 100 to 6,000 queries. Python, scikit-learn, and C++ used five timed runs per point; Weka used three because repeated JVM invocations dominate elapsed collection time.

## 3. Hardware and software environment

| Item | Value |
|---|---|
| CPU | {environment['cpu']} |
| Physical / logical cores | {environment['physical_cpu_count']} / {environment['logical_cpu_count']} |
| RAM | {environment['ram_bytes'] / 2**30:.1f} GiB |
| OS | {environment['operating_system']} |
| Python / NumPy / scikit-learn | {environment['python']} / {environment['numpy']} / {environment['scikit_learn']} |
| threadpoolctl | {environment['threadpoolctl']} |
| Java / Weka | {environment['java']} / {environment['weka']} |
| CMake / compiler | {environment['cmake']} / {environment['cpp_compiler']} |
| Native C++ flags | {environment['cpp_compile_flags']} |
| Native optimization / LTO | {environment['cpp_native_optimization']} / {environment['cpp_lto']} |
| Process affinity | logical CPUs {environment['process_affinity']} |
| Power plan | {environment['power_mode']} |
| C++ executable SHA-256 | `{environment['cpp_executable_sha256']}` |

Detected OpenBLAS and OpenMP pools reported one thread. Reliable per-run CPU frequency and temperature were unavailable, so those raw columns are intentionally blank.

## 4. Correctness verification

**Measured fact.** The fresh pre-benchmark verifier matched C++ against accepted V5.1 for 6,000/6,000 predicted classes, 6,000/6,000 positive-neighbour counts, and 6,000/6,000 vote fractions. All C++ configuration and scaling runs reproduced the expected prediction hash.

Pairwise class agreement is 99.9667% for V5.1 versus scikit-learn, 99.9833% for V5.1 versus Weka, and 100.0000% for V5.1 versus C++. See `correctness_summary.csv` and `figures/implementation_agreement_heatmap.*`.

## 5. Prediction-only performance

{_runtime_table(summary['prediction_only'])}

**Measured fact.** C++ has the lowest median at {_fmt(prediction['custom_cpp']['median_seconds'])} s, followed by scikit-learn at {_fmt(prediction['sklearn']['median_seconds'])} s and accepted V5.1 at {_fmt(prediction['custom_python_v5_1']['median_seconds'])} s. C++ is {prediction['custom_python_v5_1']['median_seconds'] / prediction['custom_cpp']['median_seconds']:.3f}x faster than V5.1 by median; scikit-learn is {prediction['custom_python_v5_1']['median_seconds'] / prediction['sklearn']['median_seconds']:.3f}x faster.

Median throughput is {throughput['custom_cpp']:.0f} queries/s for C++, {throughput['sklearn']:.0f} for scikit-learn, {throughput['custom_python_v5_1']:.0f} for V5.1, and {throughput['weka']:.0f} for Weka. Corresponding average costs are {milliseconds['custom_cpp']:.3f}, {milliseconds['sklearn']:.3f}, {milliseconds['custom_python_v5_1']:.3f}, and {milliseconds['weka']:.3f} ms/query.

## 6. Stability and outliers

**Measured fact.** The 1.5-IQR diagnostic flags {total_outliers} observations: {', '.join(f'{DISPLAY_NAMES[name]}={len(items)}' for name, items in outliers.items())}. C++ retains a {cpp_outlier['runtime_seconds']:.3f} s high observation at global order {cpp_outlier['global_run_order']}. The immediately preceding scikit-learn observation is also its only high IQR outlier, while nearby V5.1 and Weka observations remain normal.

**Interpretation.** Unlike the earlier suite, the new V5.1 and scikit-learn series stay mostly in one sustained regime; there is no monotonic start-to-finish drift. The adjacent scikit-learn and C++ highs permit a brief shared machine disturbance as one explanation, but they do not prove one. Weka also has an isolated faster run. The boxplot exposes distribution shape, and the run-order chart preserves temporal context.

**Hypothesis.** Frequency scheduling, cache state, or background activity could cause the isolated transitions. Temperature and frequency telemetry were unavailable, so thermal throttling is not claimed.

## 7. Full-pipeline performance

{_runtime_table(summary['full_pipeline'])}

**Measured fact.** C++ has the lowest five-run pipeline median at {_fmt(pipeline['custom_cpp']['median_seconds'])} s; one C++ full-pipeline run reached {_fmt(pipeline['custom_cpp']['max_seconds'])} s. Weka's {_fmt(pipeline['weka']['median_seconds'])} s total includes JVM startup and an {_fmt(fit['weka']['median_seconds'])} s internal model build.

Fit/build medians are {_fmt(fit['custom_cpp']['median_seconds'], 6)} s for C++, {_fmt(fit['sklearn']['median_seconds'], 6)} s for scikit-learn, {_fmt(fit['custom_python_v5_1']['median_seconds'], 6)} s for V5.1, and {_fmt(fit['weka']['median_seconds'])} s for Weka. These setup scopes are implementation-specific and are not perfectly equivalent.

## 8. Custom optimization history

{_history_table(summary['historical_optimization'])}

**Measured fact.** The accepted V5.1 historical median is {history_v51['speedup_vs_original']:.1f}x faster than the 33.281 s Original artifact. Historical points came from their recorded protocols, while the C++ point uses the new 20-run controlled median; the charts label this regime difference. V6 and V7 remain rejected and are excluded from the accepted history line.

## 9. C++ configuration experiments

**Measured fact.** Native heap/batch 32 was the selected report configuration at {_fmt(selected_cpp['median_seconds'])} s. Portable heap/batch 32 measured {_fmt(portable_cpp['median_seconds'])} s, so the native AVX2/LTO build was {portable_cpp['median_seconds'] / selected_cpp['median_seconds']:.2f}x faster on this machine. Native `nth_element` was rejected because it was 2.86x slower than native heap/batch 32.

**Interpretation.** The bounded heap keeps only `k=19` candidates per query and avoids materializing and partitioning the complete distance matrix. The fused native loop also reduces temporary arrays and repeated NumPy/Python passes.

## 10. Scalability with training size

| Implementation | Slope (microseconds per added train row) | R² |
|---|---:|---:|
{chr(10).join(f"| {DISPLAY_NAMES[name]} | {1e6 * train_regression[name]['slope_seconds_per_row']:.3f} | {train_regression[name]['r_squared']:.4f} |" for name in IMPLEMENTATIONS)}

**Measured fact.** Over 1,000–24,000 training rows with 1,000 fixed queries, all fitted slopes are positive. C++ has R²={train_regression['custom_cpp']['r_squared']:.4f}; V5.1 and scikit-learn have lower R² values because their results contain a visible machine-state step around 12,000–16,000 rows.

**Interpretation.** Observed scaling is approximately linear over the tested range, most clearly for C++. This empirical regression describes this dataset and machine; it does not prove formal asymptotic complexity.

## 11. Scalability with query count

| Implementation | Slope (milliseconds per added query) | R² |
|---|---:|---:|
{chr(10).join(f"| {DISPLAY_NAMES[name]} | {1000 * query_regression[name]['slope_seconds_per_row']:.4f} | {query_regression[name]['r_squared']:.4f} |" for name in IMPLEMENTATIONS)}

**Measured fact.** Query-count scaling is highly linear for V5.1 (R²={query_regression['custom_python_v5_1']['r_squared']:.4f}), scikit-learn ({query_regression['sklearn']['r_squared']:.4f}), and C++ ({query_regression['custom_cpp']['r_squared']:.4f}). Weka's lower R²={query_regression['weka']['r_squared']:.4f} reflects an anomalously fast 2,000-query point retained in the raw data.

## 12. Memory behavior

**Measured fact.** Fitted V5.1 NumPy arrays occupy {next(row['mib'] for row in summary['memory'] if row['implementation'] == 'custom_python_v5_1' and row['category'] == 'persistent_model_storage'):.3f} MiB. Estimated C++ persistent storage is {next(row['mib'] for row in summary['memory'] if row['implementation'] == 'custom_cpp' and row['category'] == 'persistent_model_storage'):.3f} MiB. Major prediction workspace is estimated at {custom_workspace['mib']:.3f} MiB for V5.1 and {cpp_workspace['mib']:.3f} MiB for native heap/batch 32.

These are array/buffer measurements and analytical estimates, not uniform process RSS peaks. They are separated by category in `memory_storage.csv`; no Python RSS value is compared with a C++ internal-buffer estimate.

## 13. Runtime versus classification quality

{_quality_table(summary['quality_metrics'])}

**Measured fact.** Accuracy, F1, and balanced accuracy differ only in the fourth decimal place among the three original implementations. C++ metrics equal V5.1 exactly because their labels and vote fractions are identical.

**Interpretation.** Implementation choice primarily changes computational cost for this experiment. The two scikit-learn disagreements and one Weka disagreement reflect implementation-specific boundary/tie or probability semantics; these tiny differences do not establish a practically more accurate model.

## 14. Rejected optimization experiments

| Experiment | Reference (s) | Candidate (s) | Decision |
|---|---:|---:|---|
{chr(10).join(f"| {row['experiment']} | {_fmt(row['reference_seconds'])} | {_fmt(row['candidate_seconds'])} | {row['decision']} |" for row in summary['rejected_experiments'])}

The full table records hypotheses, measured outcomes, and artifact-backed rejection reasons. V6 was about 1.70x slower than its paired V5.1 reference; V7 was about 1.47x slower. A smaller Python batch produced a slightly lower aggregate median in one mixed-state sweep but changed rank across trials and did not satisfy the predefined stable acceptance rule.

## 15. Limitations

- Measurements come from one Intel i5-14600KF Windows machine and one MSVC toolchain.
- CPU frequency, temperature, and background load were not locked or recorded reliably.
- Weka uses fewer scaling repetitions and fresh JVM processes; only its internal prediction timer is compared in prediction-only plots.
- Historical optimization points use their saved protocols and are not treated as one homogeneous controlled run.
- Memory values describe major arrays and buffers; they are not complete process RSS peaks.
- Simple linear fits summarize the tested range and do not establish formal complexity.

## 16. Conclusions

**Measured fact.** Experimental C++ is fastest by controlled prediction median, full-pipeline median, and throughput, and it remains exactly equivalent to V5.1 on all 6,000 outputs. Native heap/batch 32 is the strongest screened C++ configuration.

**Interpretation.** The fused loop and bounded top-k storage explain why C++ can beat the larger NumPy workspace while preserving exact exhaustive KNN behavior.

**Decision.** C++ remains an **experimental candidate**. Correctness and median performance are strong, but a high outlier persisted in both prediction-only and full-pipeline evidence, the adjacent scikit-learn anomaly does not identify the cause, and reproducibility has not yet been demonstrated across machines or repeated sessions.

## Artifact index

- Raw measurements: `raw_prediction_runs.csv`, `raw_full_pipeline_runs.csv`, `raw_scaling_train.csv`, `raw_scaling_queries.csv`, `raw_cpp_configuration.csv`
- Structured analysis: `runtime_summary.csv`, `runtime_summary.json`, and the report tables in this directory
- Environment: `environment_comprehensive.json`
- Correctness: `cpp_correctness_comprehensive.json`, `correctness_summary.csv`
- Figures: `figures/README.md` and 24 PNG/SVG figure pairs
"""
    (BENCHMARK_DIR / "benchmark_summary.md").write_text(markdown, encoding="utf-8", newline="\n")
    compact = {
        "artifact": "comprehensive KNN runtime benchmark",
        "implementations": list(IMPLEMENTATIONS),
        "prediction_runs_each": 20,
        "full_pipeline_runs_each": 5,
        "scaling_runs": {"python_sklearn_cpp": 5, "weka": 3},
        "prediction_medians_seconds": {name: prediction[name]["median_seconds"] for name in IMPLEMENTATIONS},
        "prediction_iqr_seconds": {name: prediction[name]["iqr_seconds"] for name in IMPLEMENTATIONS},
        "full_pipeline_medians_seconds": {name: pipeline[name]["median_seconds"] for name in IMPLEMENTATIONS},
        "throughput_queries_per_second": throughput,
        "iqr_outliers": outliers,
        "training_size_regression": train_regression,
        "query_size_regression": query_regression,
        "cpp_status": "experimental candidate",
        "cpp_correctness": "6000/6000 labels, positive-neighbour counts, and vote fractions exact",
        "source": "runtime_summary.json",
    }
    write_json(BENCHMARK_DIR / "benchmark_summary.json", compact)
    print(f"Comprehensive narrative -> {BENCHMARK_DIR / 'benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
