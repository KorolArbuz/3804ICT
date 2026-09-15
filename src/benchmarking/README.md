# Comprehensive runtime benchmark and visualization workflow

This package measures the current accepted Python V5.1, scikit-learn, genuine
Java/Weka IBk, and the verified experimental pure C++20 candidate. It uses the
same prepared train/test split and selected `k=19` without changing any model.

Prediction-only timings contain one complete prediction pass and exclude
prepared-file loading and model fit. Each implementation receives an explicit
warm-up, every timed implementation has at least 15 raw observations, and trial
order is randomized. Python BLAS/OpenMP pools are limited to one thread. Weka
runs in a one-CPU JVM and reports the Java classifier's internal prediction
timer. The C++ executable reports its internal exact prediction timer after an
internal warm-up.

The full-pipeline mode starts from the fixed prepared inputs and includes file
loading, model fit/build, prediction, artifact export, and metric generation.
It deliberately excludes the shared raw-data preparation, split, cross-validation,
and `k` selection because those steps are identical experiment setup rather
than implementation-specific runtime.

The comprehensive suite adds runtime stability, fit/build time, deterministic
training-size and query-count scaling, native/portable C++ configuration
screening, memory estimates, correctness/agreement, quality trade-offs, and
historical optimization evidence. Older prediction/pipeline artifacts remain
under their original `raw_*_benchmark` names.

Run every expensive measurement from the repository root after the native and
portable C++ builds exist:

```bash
python -m src.benchmarking.run_runtime_suite \
  --prediction-runs 20 --full-pipeline-runs 5 \
  --scaling-runs 5 --weka-scaling-runs 3 \
  --cpp-configuration-runs 5 \
  --cpp-native build-cpp-native/cpp_knn.exe \
  --cpp-portable build-cpp-portable/cpp_knn.exe
```

Measurement and reporting are separate. Rebuild every table, figure, and the
academic narrative without rerunning timings with:

```bash
python -m src.benchmarking.build_report_data
python -m src.benchmarking.generate_figures
python -m src.benchmarking.generate_comprehensive_summary
python -m src.benchmarking.validate_runtime_suite
```

Outputs are written under `results/runtime_benchmarks/`. CSV and JSON files
retain every raw measurement, including warm-up records and outliers. Figures
are generated only from those raw files with matplotlib and are saved as PNG
and SVG. `figures/README.md` maps every chart to its data source and suggested
report section.

The main raw files are:

- `raw_prediction_runs.csv`: 20 timed runs and one warm-up for each implementation;
- `raw_full_pipeline_runs.csv`: five complete prepared-input runs each;
- `raw_scaling_train.csv`: eight nested training prefixes;
- `raw_scaling_queries.csv`: seven nested query prefixes;
- `raw_cpp_configuration.csv`: five runs for each native/portable C++ configuration;
- `raw_memory.csv`: measured NumPy storage and explicitly labelled buffer estimates.

Weka uses three measurements per scaling point because every observation uses
a fresh JVM. CPU frequency and temperature fields remain blank when reliable
telemetry is unavailable. The full methodology, results, limitations, and C++
status are in `results/runtime_benchmarks/benchmark_summary.md`.
