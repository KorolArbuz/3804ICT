# KNN runtime benchmark summary

This suite keeps prediction-only and prepared-input full-pipeline timing separate. Every raw observation is retained; no outlier was removed or smoothed.

## Prediction-only benchmark

| Implementation | Median (s) | Min | Max | Mean | Sample std | IQR | p10 | p90 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Custom Python KNN V5.1 | 1.268500 | 0.672113 | 1.288652 | 1.193212 | 0.210269 | 0.016407 | 0.911135 | 1.281983 | Accepted V5.1, batch size 64; NumPy/BLAS limited to one thread |
| scikit-learn KNN | 0.954538 | 0.460772 | 0.956916 | 0.889079 | 0.173380 | 0.001852 | 0.658728 | 0.956549 | Brute Euclidean, uniform weights, n_jobs=1 |
| Weka IBk | 5.935960 | 3.416904 | 5.979894 | 5.770459 | 0.651535 | 0.041917 | 5.906769 | 5.969815 | Fresh one-CPU JVM per trial; internal Java prediction timer |
| Pure C++20 KNN (experimental) | 0.588488 | 0.579287 | 1.542010 | 0.670754 | 0.243364 | 0.065567 | 0.583702 | 0.669609 | Exact native Release candidate, batch size 32; internal warm-up per process |

Prediction-only timing uses 15 randomized interleaved trials after explicit warm-up. Python models use prepared in-memory arrays. C++ reports its internal prediction timer after model loading and an internal warm-up. Weka reports the existing Java prediction loop timer from a fresh one-CPU JVM for each trial.

## Full prepared-input pipeline

| Implementation | Runs | Median total (s) | Min | Max | Notes |
|---|---:|---:|---:|---:|---|
| Custom Python KNN V5.1 | 3 | 1.388410 | 0.776758 | 1.397684 | Prepared-file loading, model fit/build, prediction, artifact writing, and metrics |
| scikit-learn KNN | 3 | 1.068632 | 0.544518 | 1.079217 | Prepared-file loading, model fit/build, prediction, artifact writing, and metrics |
| Weka IBk | 3 | 14.771649 | 13.495009 | 14.777228 | Prepared-file loading, model fit/build, prediction, artifact writing, and metrics |
| Pure C++20 KNN (experimental) | 3 | 0.788644 | 0.782962 | 0.842673 | Prepared-file loading, model fit/build, prediction, artifact writing, and metrics |

The full-pipeline measurement starts from the canonical prepared NPZ, ARFF, or C++ CSV inputs. Shared raw-dataset preprocessing, train/test splitting, cross-validation, and k selection are fixed experiment setup and are not repeated per implementation.

## Historical custom optimization

| Version | Runtime (s) | Speedup vs original | Status | Main change |
|---|---:|---:|---|---|
| Original | 33.281190 | 1.000x | historical baseline | Straightforward distance calculation and full sorting |
| V1 | 1.980313 | 16.806x | superseded | Vector norms, matrix operations, and partial top-k selection |
| V2 | 1.505170 | 22.111x | superseded | Reusable workspace and one-thread row-wise selection |
| V3 | 1.419063 | 23.453x | superseded | Avoided ordered distances for uniform-vote prediction |
| V4 | 0.812958 | 40.938x | superseded | Exact pilot threshold prefilter before top-k selection |
| V5 | 0.754123 | 44.132x | superseded | Batch 64 and augmented training matrix |
| V5.1 | 1.278105 | 26.039x | accepted | Feature-major exact-fallback storage |
| C++ experimental | 0.588488 | 56.554x | experimental | Pure C++20 exhaustive fused distance and bounded-heap kernel |

## Environment

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) i5-14600KF |
| RAM | 31.8 GiB |
| OS | Windows-10-10.0.19045-SP0 |
| Python | 3.10.14 |
| NumPy | 1.26.0 |
| scikit-learn | 1.2.1 |
| Java | openjdk version "11.0.16.1" 2022-08-12 LTS |
| Weka | 3.8.6 |
| C++ compiler | MSVC _MSC_VER=1935 _MSC_FULL_VER=193532216 |
| C++ build | Release native; native/LTO=True |
| CMake | cmake version 3.25.1-msvc1 |
| BLAS | openblas 0.3.23.dev (1 thread); openblas 0.3.27 (1 thread) |
| Thread settings | BLAS/OpenMP environment variables and detected pools limited to 1 |
| Power mode | High performance (8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c) |

## Analysis

The fastest prediction-only median is **Pure C++20 KNN (experimental)** at 0.588 seconds. The fastest full prepared-input pipeline is **Pure C++20 KNN (experimental)** at 0.789 seconds.

The accepted V5.1 historical result is 26.0x faster than the saved 33.281-second original baseline. In this fresh prediction suite, scikit-learn is 1.329x faster than V5.1 by median. Earlier short runs placed both Python implementations in a faster regime; the raw results here reflect the sustained state observed during this suite.

The experimental C++ candidate is 2.156x faster than V5.1 by median and uses 0.617x the scikit-learn median. It remains experimental because the evidence comes from one native AVX2 Windows build and any raw timing outliers are retained. Its exact labels and positive-neighbour vote counts were verified against all 6,000 accepted V5.1 outputs before timing.

Run-order analysis classified machine state as **observable two regimes; no monotonic start-to-finish drift**. The largest late/early median ratio was 1.011x, so the change was not monotonic from the start to the end. Synchronized fast trials were [7, 8]; 6 IQR-rule outlier(s) were retained. Boxplots show spread and outliers, while the run-order chart shows whether changes align with execution sequence rather than implementation alone.

The fused C++ kernel reads each contiguous training row once per query batch and immediately folds exact squared distances into bounded top-k heaps. This avoids the large score matrix, NumPy/Python selection passes, and repeated temporary arrays while preserving exhaustive Euclidean semantics.

V6 and V7 are not points in the accepted optimization timeline. Their saved evidence records rejection. V6: The best exact raw-feature lower-bound candidate was about 1.70x slower than paired V5.1, far below the required stable 10% speedup. V7: The best exact block candidate achieved only a 0.6809x median paired V5.1/V7 timing ratio and was about 1.47x slower than V5.1; it did not meet the required stable 10% speedup.

Absolute runtimes depend on processor frequency, compiler, native instruction set, BLAS backend, JVM state, thermal conditions, and background load. Relative conclusions should be reproduced on the grading machine.

## Generated figures

- `figures/prediction_bar_chart.png` and `figures/prediction_bar_chart.svg`
- `figures/prediction_boxplot.png` and `figures/prediction_boxplot.svg`
- `figures/full_pipeline_bar_chart.png` and `figures/full_pipeline_bar_chart.svg`
- `figures/custom_optimization_timeline.png` and `figures/custom_optimization_timeline.svg`
- `figures/relative_speedup_chart.png` and `figures/relative_speedup_chart.svg`
- `figures/run_order_drift.png` and `figures/run_order_drift.svg`

Raw measurements are in `raw_prediction_benchmark.csv` and `raw_full_pipeline_benchmark.csv`; JSON copies retain the same rows.
