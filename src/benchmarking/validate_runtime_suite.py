"""Validate comprehensive benchmark artifacts without changing them."""

import csv
import hashlib
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.image as mpimg

from .benchmark_utils import BENCHMARK_DIR, FIGURES_DIR, IMPLEMENTATIONS, read_json


EXPECTED_CLASSIFIER_SHA256 = "7d1a675e7a0b15e4e43ff521cd2137e0481a7f58c5a70557c9c45dd2b5929652"
FIGURES = (
    "prediction_runtime_controlled",
    "prediction_runtime_distribution_comprehensive",
    "prediction_runtime_run_order_comprehensive",
    "full_pipeline_runtime_comprehensive",
    "fit_build_runtime",
    "custom_optimization_history_linear",
    "custom_optimization_history_log",
    "speedup_vs_original",
    "speedup_vs_v5_1_comprehensive",
    "v5_phase_breakdown",
    "cpp_configuration_screen",
    "runtime_vs_training_size",
    "runtime_vs_query_count",
    "prediction_throughput",
    "milliseconds_per_query",
    "runtime_vs_f1",
    "runtime_vs_accuracy",
    "confusion_matrix_v5_1",
    "cv_f1_by_k_comprehensive",
    "cv_balanced_accuracy_by_k",
    "implementation_agreement_heatmap",
    "auxiliary_memory_comparison",
    "cpp_build_comparison",
    "rejected_optimization_experiments",
)


def _rows(filename):
    with (BENCHMARK_DIR / filename).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(rows, fields):
    for row in rows:
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise AssertionError(f"Nonfinite {field} in {row}")


def main():
    prediction = _rows("raw_prediction_runs.csv")
    full = _rows("raw_full_pipeline_runs.csv")
    train = _rows("raw_scaling_train.csv")
    queries = _rows("raw_scaling_queries.csv")
    cpp = _rows("raw_cpp_configuration.csv")
    if len(prediction) != 84 or len(full) != 20 or len(train) != 176 or len(queries) != 154 or len(cpp) != 40:
        raise AssertionError("Unexpected raw artifact row count")
    timed = Counter(row["implementation"] for row in prediction if row["warmup"] == "false")
    warmups = Counter(row["implementation"] for row in prediction if row["warmup"] == "true")
    if timed != Counter({name: 20 for name in IMPLEMENTATIONS}) or warmups != Counter({name: 1 for name in IMPLEMENTATIONS}):
        raise AssertionError("Prediction run counts differ from the protocol")
    if Counter(row["implementation"] for row in full) != Counter({name: 5 for name in IMPLEMENTATIONS}):
        raise AssertionError("Full-pipeline run counts differ from the protocol")
    _finite(prediction, ("runtime_seconds", "train_rows", "test_rows", "features", "k"))
    _finite(full, ("total_seconds", "fit_seconds", "prediction_seconds", "metric_seconds"))
    _finite(train + queries, ("runtime_seconds", "train_rows", "query_rows", "queries_per_second"))
    _finite(cpp, ("runtime_seconds", "batch_size", "peak_auxiliary_bytes_estimate"))

    summary = read_json(BENCHMARK_DIR / "runtime_summary.json")
    grouped = defaultdict(list)
    for row in prediction:
        if row["warmup"] == "false":
            grouped[row["implementation"]].append(float(row["runtime_seconds"]))
    for row in summary["prediction_only"]:
        measured = statistics.median(grouped[row["implementation"]])
        if not math.isclose(measured, row["median_seconds"], rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Summary median mismatch for {row['implementation']}")

    correctness = read_json(BENCHMARK_DIR / "cpp_correctness_comprehensive.json")
    equivalence = correctness["equivalence"]
    if not correctness.get("passed") or any(
        equivalence[field] != 6000
        for field in ("prediction_matches", "positive_vote_count_matches", "positive_vote_fraction_matches")
    ):
        raise AssertionError("C++ comprehensive correctness evidence is incomplete")
    source_hash = hashlib.sha256(Path("src/custom_knn/classifier.py").read_bytes()).hexdigest()
    if source_hash != EXPECTED_CLASSIFIER_SHA256:
        raise AssertionError("Accepted V5.1 classifier source changed")

    for stem in FIGURES:
        png = FIGURES_DIR / f"{stem}.png"
        svg = FIGURES_DIR / f"{stem}.svg"
        if png.stat().st_size < 1000 or svg.stat().st_size < 1000:
            raise AssertionError(f"Empty figure: {stem}")
        pixels = mpimg.imread(png)
        if pixels.size == 0:
            raise AssertionError(f"Unreadable PNG: {png}")
        ET.parse(svg)
    markdown = (BENCHMARK_DIR / "benchmark_summary.md").read_text(encoding="utf-8")
    for heading in range(1, 17):
        if f"## {heading}." not in markdown:
            raise AssertionError(f"Missing report section {heading}")
    print(
        "Runtime suite valid: 20 timed prediction runs each; 5 full-pipeline runs each; "
        "8 train sizes; 7 query sizes; 8 C++ configurations; 24 PNG/SVG pairs; "
        "C++ exact 6000/6000; V5.1 source unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
