# Figure index

All plots are generated from saved artifacts by `python -m src.benchmarking.generate_figures`.

## `prediction_runtime_controlled.png` / `prediction_runtime_controlled.svg`

- **Shows:** controlled 6,000-query median runtime
- **Data source:** `raw_prediction_runs.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** C++ has the lowest median; Weka has the highest.

## `prediction_runtime_distribution_comprehensive.png` / `prediction_runtime_distribution_comprehensive.svg`

- **Shows:** all controlled prediction observations and outliers
- **Data source:** `raw_prediction_runs.csv`
- **Suggested report section:** Stability and outliers
- **Interpretation:** C++ retains a high implementation-specific outlier.

## `prediction_runtime_run_order_comprehensive.png` / `prediction_runtime_run_order_comprehensive.svg`

- **Shows:** actual interleaved timing sequence
- **Data source:** `raw_prediction_runs.csv`
- **Suggested report section:** Stability and outliers
- **Interpretation:** Sequence context separates shared machine state from isolated outliers.

## `full_pipeline_runtime_comprehensive.png` / `full_pipeline_runtime_comprehensive.svg`

- **Shows:** five-run prepared-input pipeline medians
- **Data source:** `raw_full_pipeline_runs.csv`
- **Suggested report section:** Full-pipeline performance
- **Interpretation:** C++ is fastest while Weka includes substantial build and JVM overhead.

## `fit_build_runtime.png` / `fit_build_runtime.svg`

- **Shows:** internal fit/build medians
- **Data source:** `raw_full_pipeline_runs.csv`
- **Suggested report section:** Full-pipeline performance
- **Interpretation:** Fit scopes differ and should be compared cautiously.

## `custom_optimization_history_linear.png` / `custom_optimization_history_linear.svg`

- **Shows:** recorded optimization history on a linear scale
- **Data source:** `historical_optimization.csv`
- **Suggested report section:** Custom optimization history
- **Interpretation:** Most runtime reduction occurred in early optimization stages.

## `custom_optimization_history_log.png` / `custom_optimization_history_log.svg`

- **Shows:** recorded optimization history on a log scale
- **Data source:** `historical_optimization.csv`
- **Suggested report section:** Custom optimization history
- **Interpretation:** The log view makes later version differences visible.

## `speedup_vs_original.png` / `speedup_vs_original.svg`

- **Shows:** recorded speedup relative to Original
- **Data source:** `historical_optimization.csv`
- **Suggested report section:** Custom optimization history
- **Interpretation:** Historical and current measurement regimes are labelled separately.

## `speedup_vs_v5_1_comprehensive.png` / `speedup_vs_v5_1_comprehensive.svg`

- **Shows:** current controlled speedup relative to V5.1
- **Data source:** `prediction_runtime_statistics.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** Values above one are faster than accepted V5.1.

## `v5_phase_breakdown.png` / `v5_phase_breakdown.svg`

- **Shows:** trusted V5 phase medians
- **Data source:** `data/processed/v5_local/final_profile.json`
- **Suggested report section:** Custom optimization history
- **Interpretation:** Matrix scoring dominates the measured V5 phases.

## `cpp_configuration_screen.png` / `cpp_configuration_screen.svg`

- **Shows:** native and portable selection/batch configurations
- **Data source:** `raw_cpp_configuration.csv`
- **Suggested report section:** C++ configuration experiments
- **Interpretation:** Native heap/batch 32 is the report configuration.

## `runtime_vs_training_size.png` / `runtime_vs_training_size.svg`

- **Shows:** runtime over nested training prefixes
- **Data source:** `raw_scaling_train.csv`
- **Suggested report section:** Training-size scalability
- **Interpretation:** Observed growth is broadly linear but has machine-state discontinuities.

## `runtime_vs_query_count.png` / `runtime_vs_query_count.svg`

- **Shows:** runtime over nested query prefixes
- **Data source:** `raw_scaling_queries.csv`
- **Suggested report section:** Query-count scalability
- **Interpretation:** Runtime generally rises with query count; Weka includes a fast-state anomaly.

## `prediction_throughput.png` / `prediction_throughput.svg`

- **Shows:** queries per second at controlled medians
- **Data source:** `prediction_runtime_statistics.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** C++ processes the most queries per second.

## `milliseconds_per_query.png` / `milliseconds_per_query.svg`

- **Shows:** average median time per query
- **Data source:** `prediction_runtime_statistics.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** Per-query cost mirrors the full-pass comparison.

## `runtime_vs_f1.png` / `runtime_vs_f1.svg`

- **Shows:** prediction median against final F1
- **Data source:** `quality_metrics.csv`
- **Suggested report section:** Runtime versus quality
- **Interpretation:** Quality is nearly unchanged while runtime varies substantially.

## `runtime_vs_accuracy.png` / `runtime_vs_accuracy.svg`

- **Shows:** prediction median against final accuracy
- **Data source:** `quality_metrics.csv`
- **Suggested report section:** Runtime versus quality
- **Interpretation:** Tiny accuracy differences should not be overinterpreted.

## `confusion_matrix_v5_1.png` / `confusion_matrix_v5_1.svg`

- **Shows:** accepted V5.1 confusion counts
- **Data source:** `confusion_counts.csv`
- **Suggested report section:** Correctness and quality
- **Interpretation:** C++ shares this matrix because its outputs are exact.

## `cv_f1_by_k_comprehensive.png` / `cv_f1_by_k_comprehensive.svg`

- **Shows:** mean CV F1 with standard-deviation bars
- **Data source:** `results/cv_results_summary.csv`
- **Suggested report section:** Model selection
- **Interpretation:** The marked k=19 is the selected parameter.

## `cv_balanced_accuracy_by_k.png` / `cv_balanced_accuracy_by_k.svg`

- **Shows:** mean CV balanced accuracy with uncertainty
- **Data source:** `results/cv_results_summary.csv`
- **Suggested report section:** Model selection
- **Interpretation:** Balanced accuracy changes little near the selected k.

## `implementation_agreement_heatmap.png` / `implementation_agreement_heatmap.svg`

- **Shows:** pairwise class agreement
- **Data source:** `correctness_summary.csv`
- **Suggested report section:** Correctness and quality
- **Interpretation:** All pairs exceed 99.96%; C++ and V5.1 are identical.

## `auxiliary_memory_comparison.png` / `auxiliary_memory_comparison.svg`

- **Shows:** comparable estimated major workspaces
- **Data source:** `memory_storage.csv`
- **Suggested report section:** Memory behavior
- **Interpretation:** The fused C++ heap uses far less algorithm workspace.

## `cpp_build_comparison.png` / `cpp_build_comparison.svg`

- **Shows:** portable versus native heap/batch-32 builds
- **Data source:** `cpp_configuration_summary.csv`
- **Suggested report section:** C++ configuration experiments
- **Interpretation:** Native AVX2/LTO materially improves this machine's runtime.

## `rejected_optimization_experiments.png` / `rejected_optimization_experiments.svg`

- **Shows:** candidate/reference runtime ratios
- **Data source:** `rejected_optimization_experiments.csv`
- **Suggested report section:** Rejected experiments
- **Interpretation:** Every candidate remains rejected under its documented acceptance evidence.
