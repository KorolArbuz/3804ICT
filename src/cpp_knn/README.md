# Exact C++20 KNN

The final adapter uses k=101, Euclidean distance, inverse-distance voting and the
frozen `balanced_low_fp` threshold from `configs/final.json`. The C++ input is the
same prepared binary64 feature matrix supplied to the other final implementations.
Training CSV row order is the original training-index order used for ties.

`knn.cpp` implements exhaustive search without BLAS, external KNN libraries,
fast-math or parallel prediction. The optimized path consumes training rows in
batches of 32 queries and maintains one bounded heap per query. Batch size 32 is
retained from the earlier implementation; it has not been retuned for k=101.
The scalar reference computes distances independently and fully sorts all rows.
An `nth_element` path is also tested against that reference.

Squared Euclidean distances rank neighbours by `(distance, training index)`.
Distance voting uses `1 / sqrt(squared_distance)`. If selected neighbours contain
exact zeros, only those neighbours vote. A score equal to the final threshold
selects class 1. The default uniform constructor preserves the baseline argmax
rule, including class 0 on equal class counts. Nonfinite input, overflowing
distances, weights or accumulated votes produce an error.

Build all C++ files together using CMake:

```text
cmake -S . -B build/cpp-portable -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/cpp-portable --config Release
ctest --test-dir build/cpp-portable -C Release --output-on-failure
```

The main `run_all.py --build` workflow uses `src.cpp_knn.bridge` to find an installed
CMake and compiler and builds under `build/final-cpp-portable`. It does not install
a toolchain. The bridge also supports multi-configuration `Release/` output paths.
Executables embed a source digest and build identity; stale or Debug executables
are rejected by the final adapter.

`-DCPP_KNN_NATIVE=ON` requests an additional local build. MSVC enables AVX2 only
after executing a CPU/OS support check. LTO is enabled only after CMake confirms
support. A native binary is specific to its target platform; portable means no
host-specific ISA selection, not an executable usable on every OS or architecture.
`-DCPP_KNN_SANITIZE=ON` enables AddressSanitizer (and UndefinedBehaviorSanitizer on
GCC/Clang). Sanitizer builds are for correctness checks, not runtime comparisons.

The CLI accepts headerless, round-trip precision feature CSVs and one binary label
per line. It writes `test_position,score_class_1,y_pred` and a JSON timing/config
record. `--help` lists options; `--identity` reports compiled source, compiler and
flags. Test labels are optional and never enter prediction.

For repeated measurement, `--persistent true --warmups 3` loads and fits once,
performs warmups, then returns a JSON `READY` line. Each stdin `PREDICT` requests a
new exhaustive prediction and returns internal elapsed seconds and a checksum.
`EXIT` closes the process. Stdout contains only protocol responses. Prediction
timing includes query validation, search, voting, score/label allocations; it
excludes process startup, loading, fit, IPC, hashing and serialization.

Current real-data checks compare all 6000 rows against the characterized manual
Python scores with an absolute tolerance of `1e-12`. Build/test evidence and any
installed-toolchain limitations are recorded in
`docs/cpp_numerical_validation.json`. The final experiment and benchmark generate
their own fresh measurements; these checks do not supply headline runtime values.
