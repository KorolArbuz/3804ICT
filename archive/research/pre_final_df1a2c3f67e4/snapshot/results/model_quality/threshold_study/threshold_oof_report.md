# Training-only threshold study

## Frozen neighbour model

Model selection already fixed **k=25, Euclidean distance, distance weighting**.
This study chooses a second-stage decision rule for that model. It does not rerun
the 64-configuration search or modify the accepted k=19/uniform baseline.

## Leakage-safe methodology

The fixed training partition supplied **24,000** out-of-fold (OOF) class-1 scores. Each training row received exactly one score from a model that did not train on it. Five-fold StratifiedKFold used shuffle=True and random_state=42, retaining the existing seed-42 80/20 split. Preprocessing was fitted separately on each fold's training rows and then applied to that fold's validation rows. The score generator was scikit-learn KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance', algorithm='brute', n_jobs=1).

The deterministic threshold grid contains **22,527** candidates: all unique OOF-score breakpoints plus 0, 0.5, 1, and nextafter(1, +infinity). The last value is the next representable float above 1 and represents the all-negative decision even when some scores equal 1. These exact breakpoints represent every attainable classification operating point; this is not an approximate evenly spaced search. Undefined metrics are retained as missing values. In particular, a rule predicting no positives has undefined precision and cannot satisfy the precision floor.

The primary objective maximizes class-1 F1, then balanced accuracy, then recall, then proximity to 0.5, then the larger threshold. The secondary objective maximizes recall among thresholds with precision >= 0.55, then F1, then balanced accuracy, then the larger threshold. The precision floor was declared before threshold selection. Both objectives use only training OOF scores; no held-out labels or metrics enter the selection.

## Selected OOF operating points

The default threshold is **0.5**.

The F1-optimized threshold is **0.279326488261**.

The recall-oriented threshold is **0.35373050977**, selected subject to OOF precision >= **0.55**.

The rule is **predict class 1 when probability_class_1 >= threshold**. A score equal to the threshold is always classified as class 1.

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

| Rule vs 0.5 | F1 delta | Recall delta | Precision delta | Balanced accuracy delta | Accuracy delta | Extra TP | FN avoided | Extra FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 optimized | +0.068915 | +0.203051 | -0.154393 | +0.047650 | -0.039000 | +1078 | +1078 | +2014 |
| Recall oriented | +0.058325 | +0.125071 | -0.093907 | +0.035116 | -0.015042 | +664 | +664 | +1025 |

## Precision/recall and confusion-count trade-off

**F1 optimized:** F1 rose (+0.068915); recall changed by **+20.31 percentage points**, and precision changed by **-15.44 percentage points**. This detected **+1078** additional defaults, avoided **+1078** false negatives, and introduced **+2014** additional false positives relative to threshold 0.5.

**Recall oriented:** F1 rose (+0.058325); recall changed by **+12.51 percentage points**, and precision changed by **-9.39 percentage points**. This detected **+664** additional defaults, avoided **+664** false negatives, and introduced **+1025** additional false positives relative to threshold 0.5.

Recall changes above are quantified in percentage points and detected defaults.
Whether they are practically material depends on the cost of missed defaults and
additional false alarms; no business costs were supplied. OOF performance helped
choose the thresholds and is not an unbiased estimate of their future performance.
The precision floor constrains training OOF selection and does not guarantee precision
on held-out or future data.

## Score-ranking metrics

For the single OOF score vector, **ROC-AUC = 0.746771**
and **average precision = 0.507291**.
These ranking metrics are reported once: changing a decision threshold leaves
the scores and their ranking unchanged.

## Freeze and limitations

The thresholds and input hashes are frozen in `threshold_selection.json` before
held-out evaluation. This report contains training OOF results only. One fixed
training partition, five folds, and threshold selection on these OOF scores limit
generalization. Threshold tuning changes the operating point, not the neighbours
or weighting. Weka thresholds and illustrative cost-sensitive thresholds were
not evaluated in this study.
