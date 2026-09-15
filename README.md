# KNN Credit Default Classification

## 1. Project Overview

This Data Mining project predicts whether a credit-card client will default on
the next month's payment. It uses K-Nearest Neighbours (KNN) as the classifier
and compares three implementations on the same train/test split:

- a manual Python and NumPy implementation;
- scikit-learn's `KNeighborsClassifier`;
- Java/Weka's `IBk` classifier.

## 2. Dataset

The experiment uses the [Default of Credit Card Clients dataset from the UCI
Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
It contains 30,000 client records and 23 original predictive attributes.

The binary target is:

- `0`: no default in the next month;
- `1`: default in the next month.

The source `ID` identifies a record and is excluded from all model features.
Download the official dataset and save it as:

```text
data/raw/default.xls
```

## 3. Implementations

### Custom Python KNN

The custom classifier manually calculates squared Euclidean distances, selects
the nearest training rows, and performs uniform voting using NumPy. It does not
use a ready-made nearest-neighbour implementation.

### scikit-learn KNN

The scikit-learn comparison uses `KNeighborsClassifier` with brute-force
Euclidean search, uniform weights, and one worker.

### Weka IBk

The Weka comparison is a genuine Java program using Weka 3.8.6 `IBk` and
`LinearNNSearch`. Weka's internal distance normalisation and internal
cross-validation are disabled so it consumes the same transformed features and
selected `k` as the Python implementations.

## 4. Experiment Methodology

- A stratified 80/20 split creates 24,000 training rows and 6,000 test rows.
- The fixed random seed is 42.
- Five-fold stratified cross-validation runs on the training partition only.
- Candidate values are the odd integers from `k=1` through `k=31`.
- Mean F1 for class 1 is the primary selection measure.
- Mean balanced accuracy and then smaller `k` resolve ties.
- The measured cross-validation result selects **k = 19**.
- `SEX`, `EDUCATION`, and `MARRIAGE` receive full one-hot encoding.
- Remaining features receive median imputation and `StandardScaler`.
- Final models use Euclidean distance and uniform voting.
- All three implementations evaluate the same untouched test rows.

## 5. Project Structure

```text
data/raw/          dataset placement instructions
src/common/        experiment settings and shared file/model helpers
src/data/          dataset loading, validation, summary, and splitting
src/preprocessing/ training-fitted transformation and compact NPZ/Weka export
src/tuning/        cross-validation and k selection
src/custom_knn/    manual NumPy KNN and its runner
src/sklearn_knn/   scikit-learn KNN runner
src/evaluation/    metrics, alignment, and implementation comparison
src/reporting/     result chart generation
weka/              Maven project for the Java/Weka implementation
results/           compact measured tables and figures
run_all.py         complete experiment entry point
```

## 6. Installation

Python 3.10 or later is recommended. From the repository root, create and
activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The Weka implementation also requires JDK 11 or later and Maven available on
`PATH`. Maven downloads the Weka dependencies during its first build.

## 7. Running the Project

Run the complete three-implementation experiment from the repository root:

```bash
python run_all.py --data data/raw/default.xls
```

Use `--skip-weka` when only the two Python implementations are required. The
main stages can also be run separately:

```bash
python -m src.data --data data/raw/default.xls
python -m src.tuning --data data/raw/default.xls
python -m src.preprocessing --data data/raw/default.xls --selected-parameters results/selected_parameters.json
python -m src.custom_knn
python -m src.sklearn_knn
python -m src.evaluation
python -m src.reporting
```

Build the Java project independently with:

```bash
mvn -f weka/pom.xml clean package
```

## 8. Results

Cross-validation selected `k=19`. The following values are from one complete
experiment run on the included environment. Each prediction time is a single
model call inside the full pipeline, so it is separate from the repeated warm
optimisation benchmark below and will vary by machine.

| Implementation | k | Accuracy | Precision | Recall | F1 | Balanced accuracy | ROC-AUC | Average precision | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom Python KNN | 19 | 0.8088 | 0.6253 | 0.3384 | 0.4391 | 0.6404 | 0.7353 | 0.4853 | 0.824 |
| scikit-learn | 19 | 0.8092 | 0.6271 | 0.3384 | 0.4395 | 0.6406 | 0.7354 | 0.4855 | 0.463 |
| Weka IBk | 19 | 0.8090 | 0.6262 | 0.3384 | 0.4393 | 0.6405 | 0.7353 | 0.4856 | 3.445 |

## Custom KNN Performance Improvement

The original custom implementation used a straightforward feature-by-feature
distance calculation and sorted every training row. The optimised version
precomputes training-vector norms, calculates exact squared Euclidean distances
with NumPy matrix operations, and uses partial top-k selection. Square roots are
only calculated when `kneighbors()` requests actual distances. It remains an
exact brute-force KNN classifier with deterministic tie handling.

The second optimisation pass reuses one distance workspace and partitions one
query row at a time, avoiding a large temporary index matrix. Its exact fallback
also uses partial selection instead of sorting all training rows. The third pass
keeps the selected neighbour set unordered for uniform-vote prediction, avoids
building selected-distance output that prediction discards, and calculates each
query norm once per batch. Public `kneighbors()` still returns ordered neighbour
indices with actual Euclidean distances.

For prediction, the third pass ranks training rows with twice the score
`0.5 * ||x||^2 - q dot x`. The omitted `||q||^2` term is constant for a query.
Numerically ambiguous boundaries still use direct sum-of-squared-differences and
deterministic training-row tie handling. The controlled checks and all 6,000
real test rows had exact class and class-1 vote-fraction agreement with v2; this
observed parity is not a proof for every possible float64 input.

The fresh same-process benchmark used `k=19`, distance batch size 128, row-wise
selection, NumPy 1.26.0 with OpenBLAS limited to one thread, one warm-up, and
five alternating measured trials. The v2 median was 1.518 seconds (range
1.502-1.545); the v3 median was 1.419 seconds (range 1.415-1.426), a 1.070x
end-to-end speedup. First calls were 1.511 and 1.410 seconds. The exact fallback
handled 38 of 6,000 queries. Phase profiling leaves row-wise partition as the
largest cost at about 0.788 seconds.

The fourth pass reduces that partition cost with a deterministic, label-free
pilot of 1,024 distinct training positions. For each query, it finds the
20th-smallest pilot score and scans the full score row, retaining every row at
or below that threshold. At least 20 pilot rows meet the threshold, so a row
above it cannot belong to the global first 20; keeping `<=` also preserves every
threshold tie. The reduced candidate array then uses the same top-k boundary
check and full-data numerical fallback as v3. All training scores are still
computed and scanned, so this remains exact exhaustive selection for the v3
score rows and retains linear search complexity.

On the real test set, the retained candidate count had median 464, 95th
percentile 656, and maximum 876 out of 24,000 rows. A 50% density guard uses the
v3 full selector when broad ties make the filtered set large. The fresh
nine-trial paired benchmark measured a v3 median of 1.411 seconds (range
1.403-1.436) and a v4 median of 0.813 seconds (range 0.808-0.826), a 1.736x
end-to-end speedup. First calls were 1.425 and 0.814 seconds. Classes and class-1
vote fractions again matched for all 6,000 rows, and the direct fallback count
remained 38. Matrix multiplication is now the largest measured phase.

The fifth pass re-benchmarked the V4 design and reduced the prediction batch
from 128 to 64. It also stores an augmented 34-column training matrix so one
matrix multiplication directly produces `||x||^2 - 2(q dot x)`, eliminating
the full score-workspace multiply and norm-add passes. The original V4
acceptance median was 0.813 seconds. In the fresh nine-trial V5 comparison, the
frozen V4 median was 0.832 seconds and V5 measured 0.754 seconds (range
0.730-0.764), a 1.103x speedup. This is a 44.132x speedup over the historical
33.281-second original baseline. NumPy/OpenBLAS remained limited to one thread,
and all 6,000 classes and vote fractions matched frozen V4 exactly.

The V5.1 pass keeps batch size 64 and stores a contiguous feature-major copy of
the training data for the 38 exact direct-fallback queries. Under the fresh
sustained-load comparison, frozen V5 measured 1.323 seconds and V5.1 measured
1.278 seconds, a 1.035x speedup. Predictions and vote fractions remained exact.

Historical and fresh summaries are in `results/custom_knn_optimization.csv`.
The historical original baseline remains labelled with an unspecified run
count rather than as a repeated-trial median.

The v4 pilot-size search, raw paired trials, selection-only timings, phase
profiles, tie-heavy density-guard check, and memory estimates are stored in
`results/custom_knn_v4_benchmark.json`.

The V5 batch-size sweep, score candidates, fallback and selection experiments,
final raw trials, phase profile, parity checks, and memory estimates are stored
in `results/custom_knn_v5_benchmark.json`.

The V5.1 small-batch and feature-major fallback measurements are stored in
`results/custom_knn_v5_1_benchmark.json`.

Production dependencies were not upgraded for this pass. A newer compatible
NumPy and scientific-Python stack can be benchmarked in an isolated environment,
with backend gains reported separately from custom-classifier changes.

Detailed values are stored in `results/metrics_comparison.csv`, while confusion
counts and pairwise agreement are in their corresponding compact CSV files.

![Cross-validation F1 by k](results/cv_f1_vs_k.png)
![Metric comparison](results/metrics_comparison.png)
![Prediction runtime comparison](results/runtime_comparison.png)

## 9. Main Source Files

- [`src/custom_knn/classifier.py`](src/custom_knn/classifier.py) contains the complete manual distance, neighbour-selection, voting, and classifier logic.
- [`src/sklearn_knn/runner.py`](src/sklearn_knn/runner.py) configures the scikit-learn comparison.
- [`weka/src/main/java/project/WekaIBkRunner.java`](weka/src/main/java/project/WekaIBkRunner.java)
  configures and runs genuine Weka `IBk`.
- [`src/preprocessing/pipeline.py`](src/preprocessing/pipeline.py) defines the training-fitted preprocessing pipeline.
- [`src/tuning/search.py`](src/tuning/search.py) performs cross-validation and selects `k`.

## 10. References

- I-Cheng Yeh, [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H),
  UCI Machine Learning Repository.
- I. Yeh and C. Lien, "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients," *Expert Systems with Applications*, 2009, [doi:10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020).
- [scikit-learn `KNeighborsClassifier` documentation](https://scikit-learn.org/1.2/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
- [Weka `IBk` documentation](https://weka.sourceforge.io/doc.stable/weka/classifiers/lazy/IBk.html).

## 11. Troubleshooting

- If the dataset is not found, confirm that `default.xls` exists under
  `data/raw/` or pass its path with `--data`.
- If Python reports a missing module, activate the virtual environment and run
  `python -m pip install -r requirements.txt`.
- If Weka cannot build, confirm that `java -version` and `mvn -version` work.
- If only Python is available, run the experiment with `--skip-weka`.
