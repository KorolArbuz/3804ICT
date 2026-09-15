# Pure C++20 exact KNN candidate

This directory contains an experimental replacement candidate for the manual
Python/NumPy KNN. It does not replace the accepted V5.1 source, scikit-learn,
or Java/Weka. The classifier and executable use only C++20 and the standard
library; Python is used separately to export the repository's canonical
processed arrays and to orchestrate verification and timing.

## Algorithm and exactness

For query `q` and training row `x`, the classifier evaluates every feature of
every training row in the original feature order:

```text
squared_distance(q, x) = sum_j (q_j - x_j)^2
```

The square root is omitted because it cannot change neighbour order. A
neighbour is ordered by `(squared_distance, original_training_index)`, so the
smaller original index resolves an exact distance tie. The optimized path
keeps a bounded maximum heap of `k` neighbours per query. A candidate replaces
the current worst heap entry only when its distance/index pair is better. It
then counts class-1 labels among the final neighbours. Uniform class
probabilities are `positive_votes / k`; equal class votes select class 0, which
matches `np.argmax` over classes `[0, 1]`.

The batch-32 loop consumes one contiguous training row at a time and updates
all active queries. It reuses transposed query and heap buffers across batches.
It remains exhaustive: no row or feature is pruned, no approximate index is
built, and the scalar and optimized paths use the same arithmetic order within
each individual distance. Compiler auto-vectorization and link-time
optimization preserve this logic under precise floating-point settings.

For `n` training rows, `m` queries, `d` features, and `k` neighbours, prediction
uses `O(mnd + mn log k)` time. The optimized working storage is
`O(bd + b + bk + m)` beyond the fitted row-major matrix and labels, where `b`
is the query batch size. The measured batch-32 CLI estimate, including returned
labels and vote counts, is 91,200 bytes. The scalar reference path uses
`O(n + m)` auxiliary elements and is retained for tests.

## Canonical input export

The C++ executable deliberately does not recreate preprocessing. Export the
existing `data/processed/train.npz` and `test.npz` arrays as round-trippable,
headerless CSV files:

```bash
python -m src.cpp_knn.export_data --output-dir data/processed/cpp
```

The generated `manifest.json` records row counts, the 33-feature order, source
NPZ hashes, and hashes of all four CSV files. All files under `data/processed/`
remain local and ignored by Git.

## Build and test

The portable release build has no native-CPU requirement:

```bash
cmake -S . -B build-cpp-portable -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp-portable --config Release
ctest --test-dir build-cpp-portable -C Release --output-on-failure
```

For a local benchmark, enable the documented native mode:

```bash
cmake -S . -B build-cpp-native -DCMAKE_BUILD_TYPE=Release -DCPP_KNN_NATIVE=ON
cmake --build build-cpp-native --config Release
ctest --test-dir build-cpp-native -C Release --output-on-failure
```

Native mode uses `-O3 -march=native` plus CMake IPO/LTO on GCC or Clang. MSVC
uses `/O2 /GL /arch:AVX2 /fp:precise` plus link-time code generation. Neither
mode enables fast-math or multithreading.

On Windows, run the following as `build-cpp-native/cpp_knn.exe`; elsewhere the
executable normally has no `.exe` suffix:

```bash
build-cpp-native/cpp_knn \
  --train-features data/processed/cpp/train_features.csv \
  --train-labels data/processed/cpp/train_labels.csv \
  --test-features data/processed/cpp/test_features.csv \
  --test-labels data/processed/cpp/test_labels.csv \
  --k 19 --batch-size 32 --selection heap \
  --warmups 1 --runs 3 \
  --output data/processed/cpp/cpp_result.json
```

The JSON contains raw prediction times, median/minimum/maximum/IQR, predictions,
positive vote counts and fractions, a deterministic FNV-1a result hash,
confusion counts, compiler/build details, and the timing boundary. Input
loading, fitting, process startup, warm-up, and JSON writing are outside the
prediction timer. Exact distance calculation, heap selection, voting, and
returned-vector construction are inside it.

Verify the real-data result against accepted V5.1, then reproduce the final
interleaved benchmark:

```bash
python -m src.cpp_knn.verify \
  --cpp-result data/processed/cpp/cpp_result.json \
  --output results/cpp_knn_correctness.json
python -m src.cpp_knn.benchmark \
  --executable build-cpp-native/cpp_knn \
  --runs 15 --output results/cpp_knn_benchmark.json
```

Use `.exe` in the benchmark command on Windows.

## Measured result

The final native benchmark ran on an Intel Core i5-14600KF under Windows with
MSVC 19.35, Python 3.10.14, NumPy 1.26.0, and scikit-learn 1.2.1. OpenBLAS and
OpenMP pools were limited to one thread. Each implementation was warmed, and
15 measured calls were randomized within paired trials.

| Implementation | Median (s) | Min (s) | Max (s) | IQR (s) |
|---|---:|---:|---:|---:|
| Pure C++20 candidate | 0.5902 | 0.5785 | 1.5921 | 0.0559 |
| Accepted Python/NumPy V5.1 | 1.2529 | 1.2361 | 1.3124 | 0.0156 |
| scikit-learn brute KNN | 0.9553 | 0.9511 | 0.9840 | 0.0056 |
| Existing Java/Weka IBk result | 3.4236 | n/a | n/a | n/a |

The raw-median ratio is 2.123x for V5.1/C++ and 0.618x for C++/scikit-learn.
This passes the 10% V5.1 improvement gate and the 1.10x scikit-learn stretch
gate on this machine. C++ was faster than V5.1 in 13 of 15 paired trials; two
C++ calls were timing outliers. CPU frequency, background load, and thermal
state were not locked.

The candidate exactly matched all 6,000 V5.1 labels, integer class-1 vote
counts, and vote fractions. Both implementations produced
`fnv1a64:28ccc6b308a3421f`, with confusion counts TN=4404, FP=269, FN=878,
TP=449. The full evidence is in `results/cpp_knn_correctness.json` and
`results/cpp_knn_benchmark.json`.

The accepted V5.1 remains the project default. The native performance evidence
comes from one AVX2 Windows machine, the standalone C++ candidate is not wired
into `run_all.py`, and the two observed outliers warrant broader reproduction
before changing the submission's primary custom implementation.

A later four-implementation benchmark suite is kept separately under
`results/runtime_benchmarks/`. Its 15-run prediction-only median was 0.5885
seconds for C++, with one retained 1.5420-second outlier. See
`src/benchmarking/README.md` for the reproducible workflow and the generated
summary for prediction-only, full-pipeline, distribution, historical, speedup,
and run-order views.
