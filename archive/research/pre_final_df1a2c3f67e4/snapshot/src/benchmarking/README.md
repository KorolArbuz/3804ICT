# Runtime benchmarking suite

This package measures accepted Python V5.1, scikit-learn, genuine Java/Weka
IBk, and the verified experimental C++20 candidate on the same prepared split
with `k=19`. It does not change any classifier or experiment setting.

Prediction-only timings cover one complete 6,000-query pass. Prepared input
loading and fitting are excluded. Trials are randomized and interleaved, Python
numerical pools use one thread, Weka reports its internal classifier timer from
a one-CPU JVM, and C++ reports its internal exact-prediction timer. All raw
timings, including outliers, remain in the statistics.

The prepared-input implementation pipeline benchmark separately includes input
loading, model fit/build, prediction, output writing, and metrics. Dataset
preparation, splitting, cross-validation, and parameter selection remain fixed
shared setup.

## Authoritative workflow

Build the native and portable C++ executables first, then collect every
expensive measurement:

```bash
python -m src.benchmarking.run_runtime_suite \
  --prediction-runs 20 \
  --full-pipeline-runs 5 \
  --scaling-runs 5 \
  --weka-scaling-runs 3 \
  --cpp-configuration-runs 5 \
  --cpp-process-mode-runs 20
```

Rebuild derived artifacts without rerunning timings:

```bash
python -m src.benchmarking.build_report_data
python -m src.benchmarking.generate_figures
python -m src.benchmarking.generate_comprehensive_summary
python -m src.benchmarking.validate_runtime_suite
```

The main outputs are under `results/runtime_benchmarks/`:

- `raw_prediction_runs.csv`: one warm-up and 20 timed observations per implementation;
- `prediction_runtime_statistics.csv`: median, spread, and deterministic bootstrap 95% intervals;
- `paired_speedups.csv` and `paired_speedup_summary.csv`: V5.1 speedups paired by trial index;
- `raw_full_pipeline_runs.csv`: five prepared-input pipeline observations per implementation;
- `raw_scaling_train.csv` and `raw_scaling_queries.csv`: deterministic scaling measurements;
- `raw_cpp_configuration.csv`: native and portable C++ configuration screen;
- `cpp_process_mode_diagnostic.csv`: 20 fresh-process and 20 persistent-process C++ timings;
- `runtime_summary.json`: machine-readable derived analysis;
- `benchmark_summary.md`: report-ready methodology, findings, and limitations;
- `figures/README.md`: chart-to-source index.

## Independent sessions

A session command creates a unique directory and refuses to overwrite it:

```bash
python -m src.benchmarking.benchmark_sessions --session-id session_02 --runs 20
python -m src.benchmarking.aggregate_sessions
```

Each completed session contains `raw_prediction_runs.csv` and
`environment.json`. Aggregation discovers completed session directories,
writes `session_summary.csv`, and creates `runtime_by_session.png` and SVG.
The repository currently contains one real session migrated from the existing
comprehensive controlled run. Two or three additional sessions should be
collected before claiming cross-session reproducibility.

## Optional CPU-affinity diagnostic

Affinity is never guessed. Select a valid logical CPU explicitly:

```bash
python -m src.benchmarking.benchmark_affinity --cpu-index 0 --runs 10
python -m src.benchmarking.generate_figures
```

The figure is created only when actual diagnostic measurements exist. No
affinity diagnostic is included in the current evidence.

Weka uses three measurements per scaling point because each observation starts
a fresh JVM. CPU frequency and temperature remain blank when reliable telemetry
is unavailable. Historical optimization points came from mixed development
sessions and are contextual evidence rather than a controlled speedup series.
