# Genuine Weka score semantics

The final adapter builds Weka 3.8.6 `IBk` from current Java source, with k=101,
inverse-distance weighting, `LinearNNSearch`, Euclidean distance with
`dontNormalize=true`, `skipIdentical=false`, and internal cross-validation disabled.
The common final transformed matrix has 60 numeric predictors. Training labels
have nominal order `{0,1}`; query classes are missing. Actual class-1 scores come
directly from `distributionForInstance`. The common decision rule is
`score >= 0.3315411365543412`.

Weka's native voting differs from the manual implementation. The loaded 3.8.6
`IBk.makeDistribution` bytecode applies `1 / (distance / sqrt(d) + 0.001)` and
starts each nominal-class accumulator at `1 / n_train`. `LinearNNSearch` includes
equal-distance neighbours at the k boundary. A zero-distance neighbour receives
finite weight 1000; positive-distance neighbours still contribute. The manual
model instead votes only among selected zero-distance neighbours when any exist,
otherwise uses exact inverse distance, and retains exactly k neighbours with the
original-training-position boundary rule.

These are intentional toolkit-native semantics. The adapter does not replace
Weka scores, recalibrate them, select a separate threshold, or use native argmax
as the final label. The common numerical threshold does not imply equal score
calibration or equal precision and recall across libraries. There is no native
`Evaluation` report presented as though it evaluated thresholded labels: the
shared Python evaluator scores the actual exported labels and probabilities.

In the initial all-row check of the current source and final matrix, all 6000
scores differed from manual Python by more than 1e-12, while 5999 labels agreed.
The largest score difference was 0.32717972755992286. At source row 28406,
manual zero-only voting produced 1.0, whereas Weka produced 0.6728202724400771.
The sole changed label was source row 14888 (test position 2945): manual score
0.3322174670079008 versus Weka 0.3294255019509184. Weka retained 102 neighbours
because the kth boundary contained two tied rows. Independent reconstruction of
the native weighting and smoothing matched six inspected cases within 3.34e-16.
These checks explain library differences; the authoritative final metrics and
pairwise diagnostic rows are regenerated from the selected run's own predictions.

The persistent worker uses one JVM for warmups and timed requests. Every
`PREDICT` performs a new search and materializes all scores and threshold labels.
Its internal timer includes query validation, copying and result allocation,
and excludes loading, fitting, JVM startup, IPC, file output and SHA-256 checksum.
KNN query evaluation is sequential; JVM service, JIT and garbage-collector
threads may still exist. Portable builds target Java 11 bytecode.

Java tests exercise actual IBk options/distributions, zero and boundary cases,
missing query classes, threshold equality, CSV paths containing spaces, and the
real persistent `READY`/`PREDICT`/`EXIT` protocol. Python rejects missing or stale
JARs using source and artifact hashes. Maven is discovered from existing tools;
the project never installs a compiler or a global toolchain automatically.
