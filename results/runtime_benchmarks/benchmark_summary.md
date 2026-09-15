# Comprehensive KNN benchmark and runtime analysis

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
| CPU | Intel(R) Core(TM) i5-14600KF |
| Physical / logical cores | 14 / 20 |
| RAM | 31.8 GiB |
| OS | Windows-10-10.0.19045-SP0 |
| Python / NumPy / scikit-learn | 3.10.14 / 1.26.0 / 1.2.1 |
| threadpoolctl | 3.5.0 |
| Java / Weka | openjdk version "11.0.16.1" 2022-08-12 LTS / 3.8.6 |
| CMake / compiler | cmake version 3.25.1-msvc1 / MSVC _MSC_VER=1935 _MSC_FULL_VER=193532216 |
| Native C++ flags | /O2 /arch:AVX2 /fp:precise; IPO/LTO enabled |
| Native optimization / LTO | True / True |
| Process affinity | logical CPUs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] |
| Power plan | High performance (8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c) |
| C++ executable SHA-256 | `52122e2d5264c0f894df409f356ff48a991fd33fcb7cc7e46d098db83f7fd34b` |

Detected OpenBLAS and OpenMP pools reported one thread. Reliable per-run CPU frequency and temperature were unavailable, so those raw columns are intentionally blank.

## 4. Correctness verification

**Measured fact.** The fresh pre-benchmark verifier matched C++ against accepted V5.1 for 6,000/6,000 predicted classes, 6,000/6,000 positive-neighbour counts, and 6,000/6,000 vote fractions. All C++ configuration and scaling runs reproduced the expected prediction hash.

Pairwise class agreement is 99.9667% for V5.1 versus scikit-learn, 99.9833% for V5.1 versus Weka, and 100.0000% for V5.1 versus C++. See `correctness_summary.csv` and `figures/implementation_agreement_heatmap.*`.

## 5. Prediction-only performance

| Implementation | Runs | Median (s) | Bootstrap 95% CI | Min | Max | IQR | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom Python KNN V5.1 | 20 | 1.2617 | [1.2488, 1.2688] | 1.2419 | 1.2872 | 0.0220 | 1.11% |
| scikit-learn KNN | 20 | 0.9550 | [0.9537, 0.9581] | 0.9511 | 0.9708 | 0.0057 | 0.52% |
| Weka IBk | 20 | 5.9477 | [5.9236, 5.9575] | 4.5962 | 5.9949 | 0.0400 | 5.15% |
| Pure C++20 KNN (experimental) | 20 | 0.6143 | [0.5859, 0.6503] | 0.5807 | 1.5586 | 0.0663 | 32.17% |

The intervals use a deterministic 10,000-resample percentile bootstrap of the raw medians. Every raw timing, including IQR outliers, remains in the calculation.

| Comparison | Trials | Median speedup | IQR | Min | Max | Candidate wins |
|---|---:|---:|---:|---:|---:|---:|
| C++20 experimental | 20 | 2.056x | 0.200 | 0.797 | 2.202 | 19/20 |
| scikit-learn | 20 | 1.319x | 0.023 | 1.280 | 1.344 | 20/20 |
| Weka IBk | 20 | 0.212x | 0.004 | 0.208 | 0.270 | 0/20 |

**Measured fact.** C++ has the lowest raw median at 0.6143 s, followed by scikit-learn at 0.9550 s and accepted V5.1 at 1.2617 s. Pairing observations by randomized trial index gives a median V5.1/C++ speedup of 2.056x with 19/20 C++ wins. The paired V5.1/scikit-learn median is 1.319x with 20/20 scikit-learn wins.

Median throughput is 9767 queries/s for C++, 6283 for scikit-learn, 4755 for V5.1, and 1009 for Weka. Corresponding average costs are 0.102, 0.159, 0.210, and 0.991 ms/query.

## 6. Stability and outliers

**Measured fact.** The 1.5-IQR diagnostic flags 3 observations: Weka IBk=1, scikit-learn KNN=1, Custom Python KNN V5.1=0, Pure C++20 KNN (experimental)=1. C++ retains a 1.559 s high observation at global order 64. The immediately preceding scikit-learn observation is also its only high IQR outlier, while nearby V5.1 and Weka observations remain normal.

**Interpretation.** Unlike the earlier suite, the new V5.1 and scikit-learn series stay mostly in one sustained regime; there is no monotonic start-to-finish drift. The adjacent scikit-learn and C++ highs permit a brief shared machine disturbance as one explanation, but they do not prove one. Weka also has an isolated faster run. The boxplot exposes distribution shape, and the run-order chart preserves temporal context.

**Hypothesis.** Frequency scheduling, cache state, or background activity could cause the isolated transitions. Temperature and frequency telemetry were unavailable, so thermal throttling is not claimed.

The targeted C++ process diagnostic retained 20 fresh-process and 20 persistent-process observations. Their medians were 0.5975 s and 1.2764 s, maxima were 1.5457 s and 1.4547 s, and the 1.5-IQR rule flagged 1 and 0 observations respectively. The persistent series shifts to a slower regime after its ninth timed observation. Process reuse therefore did not remove variability in this session; the measurement does not identify why the regime changed.

## 7. Prepared-input implementation pipeline performance

| Implementation | Runs | Median (s) | Min | Max | IQR | CV |
|---|---:|---:|---:|---:|---:|---:|
| Custom Python KNN V5.1 | 5 | 1.3966 | 1.3903 | 1.4334 | 0.0058 | 1.24% |
| scikit-learn KNN | 5 | 1.0560 | 1.0551 | 1.0620 | 0.0012 | 0.27% |
| Weka IBk | 5 | 14.8416 | 14.7998 | 14.8690 | 0.0463 | 0.20% |
| Pure C++20 KNN (experimental) | 5 | 0.8435 | 0.7803 | 1.7885 | 0.0373 | 42.30% |

**Measured fact.** C++ has the lowest five-run prepared-input pipeline median at 0.8435 s; one C++ pipeline run reached 1.7885 s. Weka's 14.8416 s total includes JVM startup and an 8.1854 s internal model build.

Fit/build medians are 0.002146 s for C++, 0.002171 s for scikit-learn, 0.011364 s for V5.1, and 8.1854 s for Weka. These setup scopes are implementation-specific and are not perfectly equivalent.

## 8. Custom optimization history

| Version | Status | Runtime (s) | Speedup vs Original | Main change |
|---|---|---:|---:|---|
| Original | Historical/accepted path | 33.2812 | 1.0x | Initial custom NumPy implementation |
| V1 | Historical/accepted path | 1.9803 | 16.8x | Matrix distances and partial selection |
| V2 | Historical/accepted path | 1.5052 | 22.1x | Workspace reuse and lower allocation overhead |
| V3 | Historical/accepted path | 1.4191 | 23.5x | Ranking-only score and less unnecessary ordering |
| V4 | Historical/accepted path | 0.8130 | 40.9x | Exact deterministic threshold prefilter |
| V5 | Historical/accepted path | 0.7541 | 44.1x | Augmented GEMM and batch 64 |
| V5.1 | Historical/accepted path | 1.2781 | 26.0x | Feature-major direct fallback layout |
| V6 | Rejected | 1.1680 | 28.5x | Partial-distance lower-bound screening |
| V7 | Rejected | 1.8406 | 18.1x | Exact block pruning |
| C++ experimental | Experimental | 0.6143 | 54.2x | Fused exact distance accumulation and bounded heap |

The history plot is contextual evidence assembled from saved development artifacts. Its points were recorded in different sessions and under their documented protocols, so they do not form a controlled speedup series and are not used for current comparative claims. V6 and V7 remain rejected and are excluded from the accepted history line.

## 9. C++ configuration experiments

**Measured fact.** Native heap/batch 32 was the selected report configuration at 0.6570 s. Portable heap/batch 32 measured 1.2549 s, so the native AVX2/LTO build was 1.91x faster on this machine. Native `nth_element` was rejected because it was 2.86x slower than native heap/batch 32.

**Interpretation.** The bounded heap keeps only `k=19` candidates per query and avoids materializing and partitioning the complete distance matrix. The fused native loop also reduces temporary arrays and repeated NumPy/Python passes.

## 10. Scalability with training size

| Implementation | Slope (microseconds per added train row) | R² |
|---|---:|---:|
| Custom Python KNN V5.1 | 8.229 | 0.9224 |
| scikit-learn KNN | 6.767 | 0.9276 |
| Weka IBk | 50.671 | 0.9494 |
| Pure C++20 KNN (experimental) | 4.221 | 0.9981 |

**Measured fact.** Over 1,000–24,000 training rows with 1,000 fixed queries, all fitted slopes are positive. C++ has R²=0.9981; V5.1 and scikit-learn have lower R² values because their results contain a visible machine-state step around 12,000–16,000 rows.

**Interpretation.** Observed scaling is approximately linear over the tested range, most clearly for C++. This empirical regression describes this dataset and machine; it does not prove formal asymptotic complexity.

## 11. Scalability with query count

| Implementation | Slope (milliseconds per added query) | R² |
|---|---:|---:|
| Custom Python KNN V5.1 | 0.2113 | 0.9998 |
| scikit-learn KNN | 0.1586 | 1.0000 |
| Weka IBk | 0.9573 | 0.9735 |
| Pure C++20 KNN (experimental) | 0.1001 | 0.9971 |

**Measured fact.** Query-count scaling is highly linear for V5.1 (R²=0.9998), scikit-learn (1.0000), and C++ (0.9971). Weka's lower R²=0.9735 reflects an anomalously fast 2,000-query point retained in the raw data.

## 12. Memory behavior

**Measured fact.** Fitted V5.1 NumPy arrays occupy 18.685 MiB. Estimated C++ persistent storage is 6.134 MiB. Major prediction workspace is estimated at 12.788 MiB for V5.1 and 0.018 MiB for native heap/batch 32.

These are array/buffer measurements and analytical estimates, not uniform process RSS peaks. They are separated by category in `raw_memory.csv`; no Python RSS value is compared with a C++ internal-buffer estimate.

## 13. Runtime versus classification quality

| Implementation | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python V5.1 | 0.8088 | 0.6253 | 0.3384 | 0.9424 | 0.4391 | 0.6404 | 0.7353 | 0.4853 |
| scikit-learn | 0.8092 | 0.6271 | 0.3384 | 0.9429 | 0.4395 | 0.6406 | 0.7354 | 0.4855 |
| Weka IBk | 0.8090 | 0.6262 | 0.3384 | 0.9426 | 0.4393 | 0.6405 | 0.7353 | 0.4856 |
| C++20 experimental | 0.8088 | 0.6253 | 0.3384 | 0.9424 | 0.4391 | 0.6404 | 0.7353 | 0.4853 |

**Measured fact.** Accuracy, F1, and balanced accuracy differ only in the fourth decimal place among the three original implementations. C++ metrics equal V5.1 exactly because their labels and vote fractions are identical.

**Interpretation.** Implementation choice primarily changes computational cost for this experiment. The two scikit-learn disagreements and one Weka disagreement reflect implementation-specific boundary/tie or probability semantics; these tiny differences do not establish a practically more accurate model.

## 14. Rejected optimization experiments

| Experiment | Reference (s) | Candidate (s) | Decision |
|---|---:|---:|---|
| Batched direct fallback | 0.7891 | 0.8231 | Rejected |
| Python batches below 64 | 1.3259 | 1.3063 | Rejected |
| V6 lower-bound pruning | 0.6870 | 1.1680 | Rejected |
| V7 block pruning | 1.2420 | 1.8406 | Rejected |
| C++ nth_element selection | 0.6570 | 1.8764 | Rejected |

The full table records hypotheses, measured outcomes, and artifact-backed rejection reasons. V6 was about 1.70x slower than its paired V5.1 reference; V7 was about 1.47x slower. A smaller Python batch produced a slightly lower aggregate median in one mixed-state sweep but changed rank across trials and did not satisfy the predefined stable acceptance rule.

## 15. Limitations

- Measurements come from one Intel i5-14600KF Windows machine and one MSVC toolchain.
- CPU frequency, temperature, and background load were not locked or recorded reliably.
- Weka uses fewer scaling repetitions and fresh JVM processes; only its internal prediction timer is compared in prediction-only plots.
- Historical optimization points use their saved protocols and are not treated as one homogeneous controlled run.
- The repository currently contains 1 real independent benchmark session(s). Two or three independently collected sessions are still needed before claiming cross-session reproducibility.
- No CPU-affinity diagnostic was collected because no logical CPU was explicitly selected; the optional command refuses to guess one.
- Memory values describe major arrays and buffers; they are not complete process RSS peaks.
- Simple linear fits summarize the tested range and do not establish formal complexity.

## 16. Conclusions

**Measured fact.** Experimental C++ has the lowest controlled prediction median, prepared-input pipeline median, and highest throughput, and it remains exactly equivalent to V5.1 on all 6,000 outputs. Its paired median speedup against V5.1 is 2.056x across 20 trials. Native heap/batch 32 is the strongest screened C++ configuration.

**Interpretation.** The fused loop and bounded top-k storage explain why C++ can beat the larger NumPy workspace while preserving exact exhaustive KNN behavior.

**Decision.** C++ remains an **experimental candidate**. Correctness and median performance are strong, but a high outlier persisted in prediction-only and prepared-input pipeline evidence, the persistent-process diagnostic entered a slower regime, and reproducibility has not yet been demonstrated across machines or repeated sessions.

## Artifact index

- Raw measurements: `raw_prediction_runs.csv`, `raw_full_pipeline_runs.csv`, `raw_scaling_train.csv`, `raw_scaling_queries.csv`, `raw_cpp_configuration.csv`, `raw_memory.csv`, `cpp_process_mode_diagnostic.csv`
- Structured analysis: `runtime_summary.csv`, `runtime_summary.json`, and the report tables in this directory
- Environment: `environment_comprehensive.json`
- Correctness: `cpp_correctness_comprehensive.json`, `correctness_summary.csv`
- Stability: `paired_speedups.csv`, `paired_speedup_summary.csv`, `session_summary.csv`, and `sessions/`
- Figures: `figures/README.md` and the compact PNG/SVG figure set
