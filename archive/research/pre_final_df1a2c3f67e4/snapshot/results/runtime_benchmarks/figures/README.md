# Figure index

All plots are generated from saved artifacts by `python -m src.benchmarking.generate_figures`.

## `prediction_runtime_controlled.png` / `prediction_runtime_controlled.svg`

- **Shows:** controlled medians and deterministic bootstrap 95% intervals
- **Data source:** `raw_prediction_runs.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** All raw timings, including outliers, contribute to the intervals.

## `prediction_runtime_distribution_comprehensive.png` / `prediction_runtime_distribution_comprehensive.svg`

- **Shows:** all controlled prediction observations and outliers
- **Data source:** `raw_prediction_runs.csv`
- **Suggested report section:** Stability and outliers
- **Interpretation:** C++ retains a high implementation-specific outlier.

## `prediction_runtime_run_order_comprehensive.png` / `prediction_runtime_run_order_comprehensive.svg`

- **Shows:** the actual interleaved timing sequence
- **Data source:** `raw_prediction_runs.csv`
- **Suggested report section:** Stability and outliers
- **Interpretation:** Sequence context helps assess shared machine-state effects.

## `paired_speedup_distribution.png` / `paired_speedup_distribution.svg`

- **Shows:** trial-paired speedup distributions relative to V5.1
- **Data source:** `paired_speedups.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** Pairing by trial index preserves shared run context; values above one favor the candidate.

## `runtime_vs_training_size.png` / `runtime_vs_training_size.svg`

- **Shows:** runtime over nested training prefixes
- **Data source:** `raw_scaling_train.csv`
- **Suggested report section:** Training-size scalability
- **Interpretation:** Observed growth is broadly linear over the tested range.

## `runtime_vs_query_count.png` / `runtime_vs_query_count.svg`

- **Shows:** runtime over nested query prefixes
- **Data source:** `raw_scaling_queries.csv`
- **Suggested report section:** Query-count scalability
- **Interpretation:** Runtime generally rises with query count.

## `cpp_configuration_screen.png` / `cpp_configuration_screen.svg`

- **Shows:** native and portable C++ selection/batch configurations
- **Data source:** `raw_cpp_configuration.csv`
- **Suggested report section:** C++ configuration experiments
- **Interpretation:** Native heap/batch 32 is the report configuration.

## `custom_optimization_history_log.png` / `custom_optimization_history_log.svg`

- **Shows:** recorded optimization history on a log scale
- **Data source:** `historical_optimization.csv`
- **Suggested report section:** Custom optimization history
- **Interpretation:** Points come from mixed sessions and are historical context, not a controlled speedup series.

## `cv_f1_by_k_comprehensive.png` / `cv_f1_by_k_comprehensive.svg`

- **Shows:** mean CV F1 with standard-deviation bars
- **Data source:** `results/cv_results_summary.csv`
- **Suggested report section:** Model selection
- **Interpretation:** The marked k=19 is the selected parameter.

## `confusion_matrix_v5_1.png` / `confusion_matrix_v5_1.svg`

- **Shows:** accepted V5.1 confusion counts
- **Data source:** `confusion_counts.csv`
- **Suggested report section:** Correctness and quality
- **Interpretation:** C++ shares this matrix because its outputs are exact.

## `implementation_agreement_heatmap.png` / `implementation_agreement_heatmap.svg`

- **Shows:** pairwise class agreement
- **Data source:** `correctness_summary.csv`
- **Suggested report section:** Correctness and quality
- **Interpretation:** All pairs exceed 99.96%; C++ and V5.1 are identical.

## `full_pipeline_runtime_comprehensive.png` / `full_pipeline_runtime_comprehensive.svg`

- **Shows:** five-run prepared-input implementation pipeline medians
- **Data source:** `raw_full_pipeline_runs.csv`
- **Suggested report section:** Prepared-input pipeline performance
- **Interpretation:** C++ is fastest while Weka includes substantial build and JVM overhead.

## `prediction_throughput.png` / `prediction_throughput.svg`

- **Shows:** queries per second at controlled medians
- **Data source:** `prediction_runtime_statistics.csv`
- **Suggested report section:** Prediction-only performance
- **Interpretation:** C++ processes the most queries per second.

## `v5_phase_breakdown.png` / `v5_phase_breakdown.svg`

- **Shows:** trusted V5 phase medians
- **Data source:** `data/processed/v5_local/final_profile.json`
- **Suggested report section:** Custom optimization history
- **Interpretation:** Matrix scoring dominates the measured V5 phases.

## `auxiliary_memory_comparison.png` / `auxiliary_memory_comparison.svg`

- **Shows:** comparable estimated major workspaces
- **Data source:** `raw_memory.csv`
- **Suggested report section:** Memory behavior
- **Interpretation:** The fused C++ heap uses less algorithm workspace.

## `rejected_optimization_experiments.png` / `rejected_optimization_experiments.svg`

- **Shows:** candidate/reference runtime ratios
- **Data source:** `rejected_optimization_experiments.csv`
- **Suggested report section:** Rejected experiments
- **Interpretation:** Every candidate remains rejected under its recorded evidence.

## `cpp_process_mode_distribution.png` / `cpp_process_mode_distribution.svg`

- **Shows:** 20 fresh-process and 20 persistent-process C++ timings
- **Data source:** `cpp_process_mode_diagnostic.csv`
- **Suggested report section:** Runtime stability
- **Interpretation:** The diagnostic tests whether process reuse explains C++ variability.

## `runtime_by_session.png` / `runtime_by_session.svg`

- **Shows:** median runtime for each completed real session
- **Data source:** `session_summary.csv`
- **Suggested report section:** Cross-session reproducibility
- **Interpretation:** More independent sessions are needed before making a reproducibility claim.
