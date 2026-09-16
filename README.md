# Credit default classification with five KNN implementations

This university project compares an original manual Python KNN baseline with four
implementations of one frozen final model: manual Python, scikit-learn, manual
C++20 and genuine Java/Weka IBk. Each implementation produces its own scores,
predictions and measured durations on the same fixed credit-default dataset.

## Dataset and protocol

Use the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients):
30,000 rows, 23 original predictors and a binary target (1 means default next month).
The original ID is retained for checking row alignment and is excluded from predictors.
See [dataset placement](data/raw/README.md); CSV, XLS and XLSX are supported.

The stratified 80/20 split uses seed 42: 24,000 training rows and 6,000 test rows.
Both partitions retain sorted source-row order. All imputation, scaling and category
learning use training rows only. The baseline has 33 transformed features; the final
representation has 60 for this dataset. No parameter search runs during the final experiment.

The test set has been inspected in earlier studies. These results establish software
correctness and reproduction on fixed data, not a new independent estimate of
generalization. The final threshold was selected using training OOF scores; the user
adopted `balanced_low_fp` after reviewing previous results. Its OOF recall constraint
does not guarantee test recall of at least 0.50.

## Frozen models

| Implementation ID | Model group | k | Vote / decision |
|---|---|---:|---|
| `baseline_python_v5_1` | Original baseline | 19 | Uniform; argmax, class 0 on ties |
| `final_python` | Final | 101 | Inverse distance; score >= 0.3315411365543412 |
| `final_sklearn` | Final | 101 | Native scikit-learn distance weights; same threshold |
| `final_cpp` | Final | 101 | Manual inverse distance; same threshold |
| `final_weka` | Final | 101 | Native Weka inverse weighting/distribution; same threshold |

Both groups use Euclidean distance. The baseline preserves its original standard
scaling, nominal encoding and batch size 64. The final representation uses signed-log
money variables, structured repayment status, engineered blocks A+B and repayment
group weight 1.0. [baseline.json](configs/baseline.json) and
[final.json](configs/final.json) record the exact settings and provenance hashes.

Manual Python and C++ order neighbours by distance then training-row position, use
only selected exact-zero neighbours when present, and compute binary64 scores.
scikit-learn uses exact brute-force search with one worker. Weka 3.8.6 uses genuine
IBk and LinearNNSearch with distance normalization, internal CV and identical-row
skipping disabled. Its native smoothing and boundary behaviour are preserved;
neither toolkit is silently forced to match manual probabilities.
See [technical notes](docs/technical_notes.md) for numerical and timing details.

## Installation

The recorded environment uses Python **3.10.14** and the exact numerical versions
in `requirements.txt`. Create a fresh virtual environment:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate`
on Linux/macOS, then run:

```text
python -m pip install -r requirements.txt
python -m pip check
```

Install a C++20 compiler and CMake, plus JDK 11 or later and Maven. The recorded
native checks use MSVC and JDK 11. The project discovers installed tools and never
installs compilers or global packages. Other toolchains require their own recorded
build/test verification; portable builds do not enable host-specific ISA flags.

## Run

From the repository root, build both native projects and run all five implementations:

```text
python run_all.py --data data/raw/UCI_Credit_Card.csv --build
```

After a successful build, omit `--build` to reuse verified current-source binaries.
Every invocation creates a new `results/final/<run_id>/`; previous runs are retained.
The main comparison uses `balanced_low_fp` and does not refit thresholds.

Collect fresh repeated measurements in the same run:

```text
python run_all.py --data data/raw/UCI_Credit_Card.csv --build --benchmark --runs 20 --warmups 3
```

The benchmark executes sequential randomized rounds, with warmups recorded separately,
and measures fresh-process prepared-input pipelines separately from resident prediction.
The baseline is a different workload; direct implementation speedups compare only the
four final implementations. Ordinary runs produce no repeated-runtime claim.

## Results

Each final run contains `run_manifest.json`, resolved configurations, data/environment
provenance, five prediction CSVs, `metrics.csv`, `agreement.csv`, row-level diagnostics,
`summary.md`, and PNG/SVG figures. A benchmark run also contains raw timings, summaries
and its timing protocol. `validation.json` distinguishes integration checks from full
release verification. A run is complete only when all five implementations finish;
numerical toolkit differences remain visible in the agreement table.

The selected completed experiment is
[`20260916T205945.220601_0000_b410e8a7`](results/final/20260916T205945.220601_0000_b410e8a7/run_manifest.json).
These values come from its independently generated
[prediction metrics](results/final/20260916T205945.220601_0000_b410e8a7/metrics.csv):

| Implementation | Accuracy | Precision | Recall | F1 | AP | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Python V5.1 baseline | 0.808833 | 0.625348 | 0.338357 | 0.439120 | 0.485337 | 0.735293 | 269 | 878 |
| Final Custom Python | 0.802000 | 0.559048 | 0.495855 | 0.525559 | 0.532645 | 0.765655 | 519 | 669 |
| Final scikit-learn | 0.802000 | 0.559048 | 0.495855 | 0.525559 | 0.532630 | 0.765646 | 519 | 669 |
| Final Custom C++20 | 0.802000 | 0.559048 | 0.495855 | 0.525559 | 0.532645 | 0.765655 | 519 | 669 |
| Final Java/Weka | 0.802167 | 0.559524 | 0.495855 | 0.525769 | 0.540097 | 0.767100 | 518 | 669 |

On all 6,000 rows, manual Python/C++ labels agree exactly and the largest score
difference is `7.77e-16`. scikit-learn has the same labels but ten score differences
above `1e-12` (maximum `0.00908349725`); its native boundary choices are retained.
Weka differs on one label, with native score differences on all rows (maximum
`0.32717972756`). The full [agreement table](results/final/20260916T205945.220601_0000_b410e8a7/agreement.csv)
and row-level diagnostics preserve these differences.

Resident prediction, twenty measured passes of 6,000 queries after three warmups:

| Final implementation | Median seconds | IQR seconds |
|---|---:|---:|
| Custom Python | 35.661570 | 0.352449 |
| scikit-learn | 1.403104 | 0.001843 |
| Custom C++20 (portable Release) | 3.677404 | 0.006887 |
| Java/Weka | 13.359275 | 0.047886 |

The separate V5.1 workload has median **1.256276 s**, IQR **0.005059 s**.
Prediction timing includes fresh score/label computation, validation and output
allocation; it excludes setup, fitting, checksum, IPC and file output. See the
[raw records](results/final/20260916T205945.220601_0000_b410e8a7/benchmark_raw.csv),
[summary](results/final/20260916T205945.220601_0000_b410e8a7/benchmark_summary.csv), and
[timing protocol](results/final/20260916T205945.220601_0000_b410e8a7/benchmark_protocol.json)
for the separately measured fresh-process scope. These are one-machine observations.
The C++ fresh-process outer median was 2.140769 s (internal prediction 1.774813 s),
below its resident prediction median despite matching artifact, configuration and
full score/label vectors. The [timing audit](docs/benchmark_timing_audit.json) found
consistent timer scopes; the cause of this process-mode difference was not measured.
Do not subtract these cross-phase medians to infer setup overhead.
Completed experiment/benchmark evidence does not by itself establish completion of
clean installation or extracted-ZIP validation; their statuses are recorded separately.
Full release verification subsequently passed all six mandatory scopes, including
fresh-source and extracted-ZIP execution with exact reproduction of all five prediction
vectors. See the selected run's [validation summary](results/final/20260916T205945.220601_0000_b410e8a7/validation.json)
and [progress record](docs/progress.md). The optional MSVC AddressSanitizer runtime was
blocked during initialization; portable/native Release and Debug CTest passed.

## Standalone commands and verification

```text
python run_all.py --data data/raw/UCI_Credit_Card.csv --only final_python
python run_all.py --data data/raw/UCI_Credit_Card.csv --build --only final_cpp final_weka
python -m src.reporting --run-dir results/final/<run_id>
python -m compileall -q src run_all.py
python -m unittest discover -s tests
python -m src.validation --data data/raw/UCI_Credit_Card.csv --full
```

`--only` intentionally records an incomplete five-way comparison and returns exit code 1;
the requested implementation's outputs remain available. Report regeneration uses only
saved scores and measurements, with no dataset loading, prediction, compiler or JVM.
The full validation command records builds, Python/C++/Java tests, all-row comparisons,
runtime evidence, clean-install reproduction, archive integrity and package checks.
Its clean child performs a finite build/test/run sequence without recursively invoking
the full validator or repeating the full benchmark.

## Project structure and submission

```text
configs/                immutable baseline and final model settings
src/data/               dataset validation and fixed split
src/preprocessing/      two training-fitted representations and canonical export
src/custom_knn/         original manual Python V5.1 baseline
src/final_knn/          manual final Python classifier
src/sklearn_knn/        scikit-learn adapter
src/cpp_knn/            manual C++20 core, CLI and build/IPC bridge
src/weka_bridge/        genuine Weka build and execution adapter
src/evaluation/         shared prediction contract, metrics and diagnostics
src/reporting/          saved-result tables and figures
src/benchmarking/       current repeated-measurement workflow
src/validation/         release verification coordinator
weka/                  Maven project, Java source and tests
tests/                 active Python and C++ tests
scripts/               source/package export, archive and clean-tree checks
results/final/<run_id>/ authoritative outputs for one final comparison
docs/                  technical notes and compact verification evidence
archive/research/      immutable historical source, tests and evidence
```

The [research archive](archive/README.md) preserves available earlier model, threshold,
operating-point and runtime studies with a byte-hash manifest. Active inference does
not read that archive. Historical unit verification is reported separately.

```text
python scripts/export_submission.py --output dist/3804ICT-final-submission.zip --run-dir results/final/<run_id>
python scripts/export_appendix.py --output dist/appendix_source_code.md
python scripts/verify_clean_checkout.py --data data/raw/UCI_Credit_Card.csv --source-zip dist/3804ICT-final-submission.zip --reference-run results/final/<run_id> --work-root verification/zip_check
```

The deterministic submission ZIP contains current working sources, tests, build
descriptions, pinned requirements, documentation, one completed run's compact evidence
and an active-source appendix. The separate `dist/3804ICT-research-history.zip` preserves
research source and evidence. Raw data, executables, third-party JARs, environments,
caches and local toolchains are excluded. Dataset placement instructions are included.
The optional clean-check `--reference-run` compares all five regenerated prediction
vectors and both prepared-matrix identities against a completed run, excluding timings.
The appendix does not replace any additional requirements set by the lecturer.

## Troubleshooting

- Missing dataset: pass its actual path with `--data`; see `data/raw/README.md`.
- Missing Python dependency: activate the intended environment, install the pinned
  requirements and run `python -m pip check`.
- Missing CMake/compiler: install a C++20 toolchain, put it on PATH, and rerun with
  `--build`. A stale executable requires rebuilding from current sources.
- Missing Java/Maven: check `java -version` and `mvn -version`; use `JAVA_HOME` if needed.
- Unavailable network for dependencies: install the pinned dependencies using your
  normal package access, then rerun in a new disposable verification directory.
- A failed or blocked native implementation leaves the comparison incomplete. Inspect
  the run manifest and validation logs before claiming the five-way result is complete.
- A verification directory already exists: choose a new `--work-root`; old evidence
  is retained rather than overwritten or silently removed.

Reference: I. Yeh and C. Lien, *The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients*, 2009,
[doi:10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020).
