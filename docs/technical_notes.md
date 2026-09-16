# Technical notes

## Frozen workload and provenance

There are two model groups. The original V5.1 baseline uses k=19, uniform voting,
33 prepared columns on this dataset, and query batch size 64. The final group uses
k=101, inverse-distance voting, 60 prepared columns and threshold
`0.3315411365543412` with an inclusive comparison. Both use the original seed-42
stratified split and sorted source-row order within the 24,000/6,000 partitions.
The four final implementations receive identical prepared binary64 values and
feature order. NPZ, 17-digit CSV and ARFF round trips are checked before prediction.
Original IDs and target values are excluded from predictor matrices. Weka query
classes are missing; only the evaluator receives test labels.

Configuration hashes cover the semantic model definition. Run manifests separately
record execution state, source hashes, the resolved configuration and implementation
outcomes. A successful compiled build is tied to current native source hashes;
neither an old executable nor a stale JAR is accepted as new-source verification.
Prepared inputs are local intermediates. Their compact manifest is retained in the
submission alongside authoritative prediction CSVs.

## Preprocessing

The baseline retains the original median imputation, numeric standardization and
full nominal encoding for SEX, EDUCATION and MARRIAGE. The migration characterization
compared both representations before and after moving source files: matrices and
feature order matched exactly. All 6,000 baseline scores, labels and row order also
matched the accepted historical `predictions_baseline_custom_v5_1.csv`. The baseline
classifier's raw SHA-256 is unchanged; see
[migration_characterization.json](migration_characterization.json).
Git text conversion is disabled for that classifier and the research snapshot so
future checkouts preserve the bytes required by these provenance checks.

For the final model, money variables use `sign(x) * log1p(abs(x))` after training-median
imputation, then training-fitted standardization. The structured PAY representation
contains positive delay and indicators for observed non-positive codes. These indicators
have neutral numeric names; no unsupported meaning is assigned to undocumented codes.
Both positive delays and indicators are standardized using training statistics. Unseen
codes create no new column during transform.

Block A contains current bill/limit, mean bill/limit, mean payment/limit,
bill-change/limit and a denominator-valid indicator. Non-positive or nonfinite limits
produce missing ratios for subsequent training-median imputation. Block B contains
positive-delay count, maximum, mean and current positive delay. A squared-distance
group weight is applied through multiplication by its square root; frozen weight
1.0 leaves the PAY/delinquency group scale unchanged. No variants are selected on test.

## Scores, ties and floating-point limits

Manual final implementations rank neighbours by computed squared Euclidean distance
and then original training-row position. For selected neighbours with distances
`d_i > 0`, the class-1 score is `sum(y_i / d_i) / sum(1 / d_i)`. If selected exact-zero
neighbours exist, they alone vote, and the score is their class-1 fraction. The final
label is 1 exactly when the score is at least the frozen threshold. Baseline uniform
voting retains class 0 on equal class counts.

C++20 performs exact exhaustive search without a third-party KNN library, BLAS,
float32, fast-math or parallel KNN prediction. The optimized bounded-heap path is
checked against an independent scalar full-sort reference; the nth-element path is
also tested. Input dimensions, finite values, overflow and invalid parameters are
checked explicitly. Portable and native Release builds have separate identities.
Native ISA flags depend on compiler/CPU support and do not make that binary portable
across machines. Correctness checks do not supply headline performance timings.

The declared absolute score tolerance for manual Python/C++ comparison is `1e-12`.
All-row checks report exact label disagreements and score differences independently.
scikit-learn's distance reductions and k-boundary selection can differ at duplicate
or tied distances. Its scores are retained. Weka's native IBk distribution includes
its own inverse-distance weighting, tie handling and smoothing; it is a toolkit
comparison, not a numerical clone of the manual kernel. Applying the common threshold
does not erase those score differences. Row-level diagnostics retain IDs, both scores,
both labels and threshold margins.

Current migration diagnostics are documented in
[cpp_numerical_validation.json](cpp_numerical_validation.json) and
[sklearn_numerical_validation.json](sklearn_numerical_validation.json). They retain
their own run/source identities. The selected final run's `agreement.csv` is the
authoritative comparison for that run. A sanitizer-runtime blocker is reported
separately from successful Release/Debug/native correctness checks.

The completed final comparison is
[`20260916T205945.220601_0000_b410e8a7`](../results/final/20260916T205945.220601_0000_b410e8a7/run_manifest.json).
Its confusion counts are:

| Implementation | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Baseline Python V5.1 | 4404 | 269 | 878 | 449 |
| Final Custom Python | 4154 | 519 | 669 | 658 |
| Final scikit-learn | 4154 | 519 | 669 | 658 |
| Final Custom C++20 | 4154 | 519 | 669 | 658 |
| Final Weka | 4155 | 518 | 669 | 658 |

Python/C++ maximum absolute score difference is `7.771561172376096e-16`, with zero
differences above `1e-12` and zero label disagreements. scikit-learn differs from
manual Python on ten scores above tolerance, maximum `0.009083497248437489`, with no
label disagreements at the frozen threshold. Weka differs on one label and all
6,000 scores, maximum `0.32717972755992286`. The separate
[Weka semantics analysis](weka_semantics.md) explains its native behaviour. None of
these external-library outputs was replaced by manual scores to improve agreement.

## Runtime scopes

The resident prediction benchmark starts from loaded prepared data and a fitted model.
Each command computes fresh neighbours, scores and labels. Its timer includes query
validation and output allocation/materialization, and ends before checksum calculation,
IPC serialization and file output. Python numerical libraries use one compute worker;
C++ prediction is sequential; Weka executes sequential IBk queries even though JVM
service, JIT and GC threads may exist. Fit/setup durations are recorded separately.

Each implementation receives three warmups and twenty measured trials in randomized
interleaved order by default. All samples, including outliers, remain in the raw CSV.
Prediction summaries report median, quartiles/IQR and additional distribution statistics.
The fresh-process prepared pipeline is measured separately for five trials: process
startup, loading prepared inputs, fitting, prediction and CSV output are included;
raw-data preprocessing and builds are excluded. Python/C++ load CSV and Weka loads ARFF,
so that broader scope includes toolkit-specific ingestion costs.

No heavy correctness test or independent workload should run concurrently with headline
measurements. Timings are observations on one machine. V5.1 has a different representation,
k and voting rule, so its duration is shown separately; implementation speedup ratios
apply only within the four-model final group.

For the selected run, the resident prediction medians/IQRs (seconds) are Custom
Python `35.661570 / 0.352449`, scikit-learn `1.403104 / 0.001843`, portable C++
`3.677404 / 0.006887`, and Weka `13.359275 / 0.047886`. The distinct baseline workload
is `1.256276 / 0.005059`. There are twenty measured resident samples per implementation.
The median paired Custom-Python/implementation ratios are `25.397863` for scikit-learn,
`9.686462` for C++ and `2.667060` for Weka; pairs share run, phase, trial and workload
identities. These are medians of per-trial ratios, not ratios of separately computed medians.

The five fresh-process prepared-pipeline medians/IQRs are Custom Python
`32.378449 / 0.428924`, scikit-learn `2.090856 / 0.014041`, C++ `2.140769 / 0.008043`,
and Weka `22.506577 / 0.037660`; the distinct baseline is `1.884914 / 0.040399`.
These scopes use different process lifecycles. Subtracting their medians from resident
medians would not isolate setup cost. All 140 raw observations, including warmups,
remain in the selected run's benchmark CSV; warmups are excluded from the summaries.

The C++ fresh-process internal prediction median was `1.7748134` s, versus resident
`3.6774041` s, despite the same compiled artifact, configuration, workload and full
output vectors. Its fresh outer median `2.140768900047988` s exceeds its own internal
prediction scope, as expected. The [saved-evidence timing audit](benchmark_timing_audit.json)
found no timer-scope or model substitution error. Process placement, frequency,
temperature and memory/cache behaviour were not sampled, so the cause of the stable
process-mode difference is unknown. Every original sample remains intact.

Raw `batch_size` identifies the adapter's reported unit: V5.1 uses 64 and the C++
query batch is 32. scikit-learn's recorded value 1 does not describe its internal
library blocking, which is not exposed or claimed to be one query per batch.

## Reporting, validation and packages

Reporting reads saved predictions, metrics, agreement and raw benchmark CSVs. It checks
configuration identities and recalculates metrics/disagreements before generating tables.
It uses no raw dataset, inference, JVM or compiler. Without repeated benchmark samples,
runtime figures are explicitly omitted. Figures use deterministic PNG/SVG output metadata.
Precision-recall curves compare baseline and final manual scores; complete toolkit
comparisons remain available in tables and row-level diagnostics.
The evaluator uses the explicit scikit-learn `zero_division=0` convention for
precision, recall, F1 and F0.5 when their denominator is zero. This convention does
not affect any of the five reported rows. Both target classes are required for
the ranking-metric comparison; undefined ROC-AUC is not replaced by a zero score.

Active release checks cover prediction contracts, builds, cross-language correctness,
all fixed rows, numerical toolkit differences, runtime evidence and packaging.
Historical tests run only after restoring the snapshot to a separate disposable tree;
the original archive bytes are verified before and after. Their results are distinct
from active release verification. Full validation must not recursively call itself
through the clean-install check or silently count skipped native implementations as passes.

The clean-tree helper creates a fresh environment, installs the unchanged pinned
requirements, runs `pip check`, tests and a new five-model build/integration run.
The ZIP check actually extracts the current working-tree export and repeats this
finite sequence. ZIP inventory verification and actual execution are separate records.
With `--reference-run`, the clean check also compares all five regenerated score and
label vectors with a completed parent run: row identities, ground truth and labels
must agree, scores must agree within the declared `1e-12` absolute tolerance, and
both groups' canonical matrix hashes, feature order and configuration identities
must match. Runtime measurements are excluded from this reproduction comparison.
The package omits personal absolute paths rather than modifying evidence bytes to
hide them. SHA-256 manifests cover the included bytes. Raw data and build products are
provided/regenerated separately; historical source is delivered in its own archive.

These checks establish program behaviour and fixed-data reproduction. Repeated prior
inspection of the test prevents describing it as untouched or independently selected
for this final operating point. Absence of a significant difference does not establish
equivalence, and the training OOF recall criterion does not guarantee test recall.
