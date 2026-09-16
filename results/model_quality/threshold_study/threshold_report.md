# Frozen-model decision-threshold study

## 1. Objective

Assess whether a decision threshold selected only from training OOF scores improves
class-1/default F1 and recall for the already-selected KNN. All changes below are
observed on the fixed split; thresholds were not chosen from test performance.

## 2. Frozen model

The neighbour model is **k=25, Euclidean distance, distance weighting**. The accepted
V5.1 reference remains **k=19, Euclidean, uniform, default argmax**. Neither model
selection nor the accepted implementation was modified for this threshold study.

## 3. Leakage-safe OOF methodology

The fixed training partition supplied **24,000** out-of-fold (OOF) class-1 scores. Each training row received exactly one score from a model that did not train on it. Five-fold StratifiedKFold used shuffle=True and random_state=42, retaining the existing seed-42 80/20 split. Preprocessing was fitted separately on each fold's training rows and then applied to that fold's validation rows. The score generator was scikit-learn KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance', algorithm='brute', n_jobs=1).

The deterministic threshold grid contains **22,527** candidates: all unique OOF-score breakpoints plus 0, 0.5, 1, and nextafter(1, +infinity). The last value is the next representable float above 1 and represents the all-negative decision even when some scores equal 1. These exact breakpoints represent every attainable classification operating point; this is not an approximate evenly spaced search. Undefined metrics are retained as missing values. In particular, a rule predicting no positives has undefined precision and cannot satisfy the precision floor.

The primary objective maximizes class-1 F1, then balanced accuracy, then recall, then proximity to 0.5, then the larger threshold. The secondary objective maximizes recall among thresholds with precision >= 0.55, then F1, then balanced accuracy, then the larger threshold. The precision floor was declared before threshold selection. Both objectives use only training OOF scores; no held-out labels or metrics enter the selection.

The threshold selection and relevant input hashes were frozen before final
evaluation. The OOF report and figures were generated before the test evaluation.
The test partition was excluded from second-stage threshold selection; its
threshold-0.5 result had already been reported by the earlier model-quality study.

## 4. OOF threshold results

| Rule | Threshold | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected model, threshold 0.5 | 0.5 | 0.812667 | 0.643996 | 0.342437 | 0.946231 | 0.447122 | 0.644334 |
| Selected model, F1 threshold | 0.279326488261 | 0.773667 | 0.489603 | 0.545489 | 0.838478 | 0.516037 | 0.691984 |
| Selected model, recall threshold | 0.35373050977 | 0.797625 | 0.550089 | 0.467508 | 0.891392 | 0.505448 | 0.679450 |

| Rule | TN | FP | FN | TP | Predicted-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected model, threshold 0.5 | 17686 | 1005 | 3491 | 1818 | 0.117625 |
| Selected model, F1 threshold | 15672 | 3019 | 2413 | 2896 | 0.246458 |
| Selected model, recall threshold | 16661 | 2030 | 2827 | 2482 | 0.188000 |

The single OOF score vector had ROC-AUC **0.746771**
and average precision **0.507291**.
These ranking metrics do not change with the threshold.

## 5. Frozen thresholds

The default threshold is **0.5**.

The F1-optimized threshold is **0.279326488261**.

The recall-oriented threshold is **0.35373050977**, selected subject to OOF precision >= **0.55**.

The rule is **predict class 1 when probability_class_1 >= threshold**. A score equal to the threshold is always classified as class 1.

The OOF precision floor is a selection constraint, not a guarantee of test or future
precision. Ordinary commands refuse to overwrite a completed study. An explicit
`--verify-rerun` checks frozen hashes and thresholds and repeats evaluation without
selecting a new threshold.

## 6. Untouched-test results

Preprocessing was fitted once to all 24,000 training rows. The primary score vector
comes from EnhancedCustomKNNClassifier with the frozen model configuration. The
three decision rules use that same vector and the thresholds fixed above.

| Rule | Threshold | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted V5.1: k=19, uniform, argmax | argmax | 0.808833 | 0.625348 | 0.338357 | 0.942435 | 0.439120 | 0.640396 |
| Selected model, threshold 0.5 | 0.5 | 0.811833 | 0.640227 | 0.340618 | 0.945645 | 0.444663 | 0.643132 |
| Selected model, F1 threshold | 0.279326488261 | 0.768833 | 0.480341 | 0.552374 | 0.830302 | 0.513845 | 0.691338 |
| Selected model, recall threshold | 0.35373050977 | 0.795833 | 0.544894 | 0.466466 | 0.889364 | 0.502639 | 0.677915 |

| Rule | TN | FP | FN | TP | Predicted-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted V5.1: k=19, uniform, argmax | 4404 | 269 | 878 | 449 | 0.119667 |
| Selected model, threshold 0.5 | 4419 | 254 | 875 | 452 | 0.117667 |
| Selected model, F1 threshold | 3880 | 793 | 594 | 733 | 0.254333 |
| Selected model, recall threshold | 4156 | 517 | 708 | 619 | 0.189333 |

For the one Enhanced Custom test score vector, **ROC-AUC = 0.739541**
and **average precision = 0.503025**. Threshold
tuning changes neither score-ranking metric. The previously reported default
threshold result is retained as the reference for all threshold-only deltas.

## 7. Model-change vs threshold-change decomposition

| Change | F1 delta | Recall delta | Precision delta | Balanced accuracy delta |
| --- | ---: | ---: | ---: | ---: |
| Model configuration: k=19 uniform to k=25 distance at default rule | +0.005543 | +0.002261 | +0.014878 | +0.002735 |
| Threshold only: 0.5 to F1 optimized | +0.069182 | +0.211756 | -0.159886 | +0.048206 |
| Threshold only: 0.5 to Recall oriented | +0.057976 | +0.125848 | -0.095332 | +0.034784 |

The first row compares different neighbour configurations at their default rules.
The remaining rows change only the selected model's operating point. These sources
of change are reported separately.

## 8. Precision/recall trade-off

**F1 optimized:** F1 rose (+0.069182); recall changed by **+21.18 percentage points**, and precision changed by **-15.99 percentage points**. This detected **+281** additional defaults, avoided **+281** false negatives, and introduced **+539** additional false positives relative to threshold 0.5.

**Recall oriented:** F1 rose (+0.057976); recall changed by **+12.58 percentage points**, and precision changed by **-9.53 percentage points**. This detected **+167** additional defaults, avoided **+167** false negatives, and introduced **+263** additional false positives relative to threshold 0.5.

An OOF-selected precision floor need not hold on the test set. A recall gain can
come with lower precision and lower accuracy; this is not a universal improvement.

## 9. Confusion-matrix changes

| Rule vs 0.5 | F1 delta | Recall delta | Precision delta | Balanced accuracy delta | Accuracy delta | Extra TP | FN avoided | Extra FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 optimized | +0.069182 | +0.211756 | -0.159886 | +0.048206 | -0.043000 | +281 | +281 | +539 |
| Recall oriented | +0.057976 | +0.125848 | -0.095332 | +0.034784 | -0.016000 | +167 | +167 | +263 |

All differences are candidate minus selected-model threshold 0.5, except FN avoided,
which is reference FN minus candidate FN. Positive extra TP counts additional
defaults detected; positive extra FP counts additional non-defaults flagged.

## 10. Bootstrap intervals

The paired bootstrap resampled the same test-row indices for both decision rules:
10,000 resamples with seed 20260916. Each threshold candidate is compared with the selected model at
0.5. The table reports 95% percentile intervals and explicitly states whether each
interval includes zero. Count deltas refer to resamples of the original test size.

| Rule vs 0.5 | Metric | Observed delta | 95% percentile interval | Zero comparison | Valid resamples |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 optimized | f1 | +0.069182 | [+0.047985, +0.091602] | Excludes zero | 10000 |
| F1 optimized | recall | +0.211756 | [+0.190511, +0.233757] | Excludes zero | 10000 |
| F1 optimized | precision | -0.159886 | [-0.186499, -0.133126] | Excludes zero | 10000 |
| F1 optimized | balanced_accuracy | +0.048206 | [+0.036777, +0.059966] | Excludes zero | 10000 |
| F1 optimized | TP | +281.000000 | [+251.000000, +314.000000] | Excludes zero | 10000 |
| F1 optimized | FP | +539.000000 | [+496.000000, +583.000000] | Excludes zero | 10000 |
| F1 optimized | FN | -281.000000 | [-314.000000, -251.000000] | Excludes zero | 10000 |
| Recall oriented | f1 | +0.057976 | [+0.040179, +0.076083] | Excludes zero | 10000 |
| Recall oriented | recall | +0.125848 | [+0.108171, +0.143825] | Excludes zero | 10000 |
| Recall oriented | precision | -0.095332 | [-0.118559, -0.072162] | Excludes zero | 10000 |
| Recall oriented | balanced_accuracy | +0.034784 | [+0.025367, +0.044348] | Excludes zero | 10000 |
| Recall oriented | TP | +167.000000 | [+143.000000, +192.000000] | Excludes zero | 10000 |
| Recall oriented | FP | +263.000000 | [+232.000000, +293.000000] | Excludes zero | 10000 |
| Recall oriented | FN | -167.000000 | [-192.000000, -143.000000] | Excludes zero | 10000 |

These are conditional resampling intervals for this fixed split and frozen model.
They are not automatically labelled statistically significant and do not account
for training-data variation, model selection, threshold selection, or distribution
shift. No threshold was adjusted in response to these intervals.

## 11. Probability-semantics caveat

Enhanced Custom is the primary implementation; scikit-learn scores provide a
diagnostic comparison. The earlier model-quality evaluation identified seven
probability-difference rows associated with exact-zero handling, boundary ties,
and floating-point distance arithmetic. The new diagnostic evidence below checks
the default and both selected threshold rules, including whether a selected
threshold falls between the differing scores. Row identifiers and probabilities
for any label disagreements are retained in the study diagnostics.

Across **6,000** test rows, **7** score differences exceeded 1e-10. The maximum absolute score difference was **0.0286524932567**.

| Rule | Threshold | Label disagreements | Threshold separates scores | Disagreement row IDs |
| --- | ---: | ---: | ---: | ---: |
| Default 0.5 | 0.5 | 0 | 0 | None |
| F1 optimized | 0.279326488261 | 0 | 0 | None |
| Recall oriented | 0.35373050977 | 0 | 0 | None |

The `threshold_score_diagnostics.csv` file gives affected-row identifiers, both probabilities, causes, thresholded labels, and each threshold's distance to the interval between the two scores. The full agreement and provenance records are in `threshold_consistency.json`.

Enhanced Custom repeatable: **True**. Default-rule differences from the original selected-model labels: **0**; scores exactly equal to 0.5: **0**.

Weka numeric thresholds were **not evaluated**. Genuine Weka IBk has native
inverse-distance and nominal-probability semantics, so the same numeric threshold
must not be assumed equivalent. Applying these thresholds to Weka would require
a separately labelled exploratory appendix.

## 12. Optional cost-sensitive appendix

Not run. Business costs were not supplied. An illustrative training-OOF analysis
of FP + ratio * FN at FN/FP ratios 1, 2, 5 and 10 could be studied separately; it
would not replace the primary F1 objective or justify changing a frozen threshold
after test inspection.

## 13. Limitations

Only one fixed dataset split and five OOF folds were used. The best OOF thresholds
were selected using the same OOF scores summarized here, so OOF gains can be
optimistic. The test partition had previously been evaluated at the default rule,
but no test scores, labels or metrics were used to choose these thresholds. A
single held-out sample and a paired bootstrap do not establish external validity.
The OOF precision floor is not a population guarantee. Probability estimates can
differ slightly between exact implementations near zero distances and boundary
ties. Neither probability calibration nor business costs were established.

## 14. Conclusion

**F1 optimized:** F1 rose (+0.069182); recall changed by **+21.18 percentage points**, and precision changed by **-15.99 percentage points**. This detected **+281** additional defaults, avoided **+281** false negatives, and introduced **+539** additional false positives relative to threshold 0.5.

**Recall oriented:** F1 rose (+0.057976); recall changed by **+12.58 percentage points**, and precision changed by **-9.53 percentage points**. This detected **+167** additional defaults, avoided **+167** false negatives, and introduced **+263** additional false positives relative to threshold 0.5.

For f1 optimized, the threshold-only F1 change (+0.069182) was larger than the earlier model-configuration change (+0.005543); the recall change (+0.211756) was larger than the model-configuration recall change (+0.002261). These comparisons describe usefulness for F1 and default detection on this fixed split; they do not establish an overall business benefit.

For recall oriented, the threshold-only F1 change (+0.057976) was larger than the earlier model-configuration change (+0.005543); the recall change (+0.125848) was larger than the model-configuration recall change (+0.002261). These comparisons describe usefulness for F1 and default detection on this fixed split; they do not establish an overall business benefit.

The f1 optimized paired bootstrap intervals exclude zero for f1, recall, precision, balanced_accuracy. Intervals containing zero leave the direction of those changes uncertain under this resampling. Intervals excluding zero support a consistent direction within the observed test sample, not robustness to new populations.

The recall oriented paired bootstrap intervals exclude zero for f1, recall, precision, balanced_accuracy. Intervals containing zero leave the direction of those changes uncertain under this resampling. Intervals excluding zero support a consistent direction within the observed test sample, not robustness to new populations.

Recall gains are quantified above; their practical materiality depends on the extra false positives and unspecified business costs. Threshold tuning changes the precision/recall operating point and leaves ROC-AUC and average precision unchanged. The evidence does not make the model universally better.
