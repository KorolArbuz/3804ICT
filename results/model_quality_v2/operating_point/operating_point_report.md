# Final V2 operating-point optimization

## 1. Objective

This study asks whether the frozen final V2 KNN can operate with fewer false positives while preserving predeclared minimum recall, F1, or accuracy. It changes only the score threshold; it does not change the neighbour model or score ranking.

## 2. Frozen V2 model

The model remains k=101, Euclidean, inverse-distance voting, signed-log money features, structured PAY representation, Blocks A+B, and PAY/delinquency weight 1.0. Its configuration hash was verified before OOF scoring and again before legacy evaluation.

## 3. Training-only OOF protocol

Five-fold `StratifiedKFold(shuffle=True, random_state=42)` generated one held-out score for each of 24,000 training rows. Every transform was fitted inside its fold. Threshold objectives, constraints, and tie-breaks were predeclared, and all six thresholds were frozen before the legacy test was scored.

## 4. Threshold frontier

The exact frontier contains 23873 threshold candidates, including every unique OOF score, fixed boundary thresholds, and the frozen current threshold. The three-dimensional FP/recall/F1 Pareto set contains 3256 points; the min-FP-by-recall frontier contains 5296 points. Shared OOF score ranking was AP 0.536350 and ROC-AUC 0.775402.

## 5. Frozen operating points

See `operating_points.json` and `oof_operating_point_report.md` for the immutable training-OOF thresholds and metrics. No threshold was selected from legacy labels. The predeclared objectives that converged to the same frozen threshold were: min_fp_recall_50, max_precision_recall_50.

## 6. OOF trade-offs

The OOF tables show that FP reduction generally exchanges true positives for true negatives. A lower-FP operating point is appropriate only for the constraint that defined it; it is not a universal improvement.

## 7. Legacy comparison

**Legacy held-out test comparison; this test has prior analyst exposure and is not independent validation for operating-point selection.** The same deterministic manual-kernel continuous score vector was used for every threshold. Its shared AP was 0.532645 and ROC-AUC was 0.765655.

| operating_point         |   threshold |   accuracy |   precision |   recall |   specificity |       f1 |     f0_5 |   balanced_accuracy |   TN |   FP |   FN |   TP |
|:------------------------|------------:|-----------:|------------:|---------:|--------------:|---------:|---------:|--------------------:|-----:|-----:|-----:|-----:|
| current_v2_threshold    |    0.280150 |   0.787333 |    0.518600 | 0.535795 |      0.858763 | 0.527057 | 0.521950 |            0.697279 | 4013 |  660 |  616 |  711 |
| min_fp_recall_50        |    0.331541 |   0.802000 |    0.559048 | 0.495855 |      0.888936 | 0.525559 | 0.545153 |            0.692396 | 4154 |  519 |  669 |  658 |
| min_fp_f1_52            |    0.382471 |   0.808333 |    0.587364 | 0.448380 |      0.910550 | 0.508547 | 0.553077 |            0.679465 | 4255 |  418 |  732 |  595 |
| min_fp_accuracy_guard   |    0.356009 |   0.805333 |    0.572736 | 0.471741 |      0.900064 | 0.517355 | 0.549219 |            0.685902 | 4206 |  467 |  701 |  626 |
| max_precision_recall_50 |    0.331541 |   0.802000 |    0.559048 | 0.495855 |      0.888936 | 0.525559 | 0.545153 |            0.692396 | 4154 |  519 |  669 |  658 |
| f0_5_optimized          |    0.440138 |   0.812833 |    0.614607 | 0.412208 |      0.926600 | 0.493460 | 0.559648 |            0.669404 | 4330 |  343 |  780 |  547 |

## 8. FP avoided versus TP lost

| operating_point         |   FP_avoided |   TP_lost |   additional_FN |   TN_gained |   net_correct_classification_change |   FP_avoided_per_TP_lost |   TP_lost_per_100_FP_avoided |   accuracy_delta |   precision_delta |   recall_delta |   f1_delta |   f0_5_delta |
|:------------------------|-------------:|----------:|----------------:|------------:|------------------------------------:|-------------------------:|-----------------------------:|-----------------:|------------------:|---------------:|-----------:|-------------:|
| current_v2_threshold    |            0 |         0 |               0 |           0 |                                   0 |               nan        |                   nan        |         0.000000 |          0.000000 |       0.000000 |   0.000000 |     0.000000 |
| min_fp_recall_50        |          141 |        53 |              53 |         141 |                                  88 |                 2.660377 |                    37.588652 |         0.014667 |          0.040449 |      -0.039940 |  -0.001498 |     0.023203 |
| min_fp_f1_52            |          242 |       116 |             116 |         242 |                                 126 |                 2.086207 |                    47.933884 |         0.021000 |          0.068765 |      -0.087415 |  -0.018510 |     0.031127 |
| min_fp_accuracy_guard   |          193 |        85 |              85 |         193 |                                 108 |                 2.270588 |                    44.041451 |         0.018000 |          0.054136 |      -0.064054 |  -0.009702 |     0.027269 |
| max_precision_recall_50 |          141 |        53 |              53 |         141 |                                  88 |                 2.660377 |                    37.588652 |         0.014667 |          0.040449 |      -0.039940 |  -0.001498 |     0.023203 |
| f0_5_optimized          |          317 |       164 |             164 |         317 |                                 153 |                 1.932927 |                    51.735016 |         0.025500 |          0.096007 |      -0.123587 |  -0.033597 |     0.037698 |

The largest observed FP reduction came from `f0_5_optimized`: 317 FP avoided for 164 TP lost, or 51.735 TP lost per 100 FP avoided. This ratio is descriptive and is not a business-cost model.

## 9. Paired bootstrap

Intervals below are deterministic 95% percentile intervals from 10,000 paired row resamples. Differences are candidate minus current threshold; `includes_zero` is descriptive and is not automatically labelled statistical significance.

| operating_point         | metric            |   observed_difference |   ci_lower_95 |   ci_upper_95 | includes_zero   |
|:------------------------|:------------------|----------------------:|--------------:|--------------:|:----------------|
| min_fp_recall_50        | FP                |           -141.000000 |   -164.000000 |   -119.000000 | False           |
| min_fp_recall_50        | TP                |            -53.000000 |    -67.000000 |    -39.000000 | False           |
| min_fp_recall_50        | FN                |             53.000000 |     39.000000 |     67.000000 | False           |
| min_fp_recall_50        | accuracy          |              0.014667 |      0.010333 |      0.019333 | False           |
| min_fp_recall_50        | precision         |              0.040449 |      0.029666 |      0.051591 | False           |
| min_fp_recall_50        | recall            |             -0.039940 |     -0.050769 |     -0.029629 | False           |
| min_fp_recall_50        | f1                |             -0.001498 |     -0.011050 |      0.007688 | True            |
| min_fp_recall_50        | f0_5              |              0.023203 |      0.013459 |      0.033134 | False           |
| min_fp_recall_50        | balanced_accuracy |             -0.004883 |     -0.010882 |      0.000830 | True            |
| min_fp_f1_52            | FP                |           -242.000000 |   -272.000000 |   -212.000000 | False           |
| min_fp_f1_52            | TP                |           -116.000000 |   -137.000000 |    -96.000000 | False           |
| min_fp_f1_52            | FN                |            116.000000 |     96.000000 |    137.000000 | False           |
| min_fp_f1_52            | accuracy          |              0.021000 |      0.014833 |      0.027167 | False           |
| min_fp_f1_52            | precision         |              0.068765 |      0.053216 |      0.084682 | False           |
| min_fp_f1_52            | recall            |             -0.087415 |     -0.102853 |     -0.072409 | False           |
| min_fp_f1_52            | f1                |             -0.018510 |     -0.032414 |     -0.004942 | False           |
| min_fp_f1_52            | f0_5              |              0.031127 |      0.017066 |      0.045243 | False           |
| min_fp_f1_52            | balanced_accuracy |             -0.017814 |     -0.025974 |     -0.009672 | False           |
| min_fp_accuracy_guard   | FP                |           -193.000000 |   -220.025000 |   -167.000000 | False           |
| min_fp_accuracy_guard   | TP                |            -85.000000 |   -103.000000 |    -68.000000 | False           |
| min_fp_accuracy_guard   | FN                |             85.000000 |     68.000000 |    103.000000 | False           |
| min_fp_accuracy_guard   | accuracy          |              0.018000 |      0.012500 |      0.023500 | False           |
| min_fp_accuracy_guard   | precision         |              0.054136 |      0.040913 |      0.067794 | False           |
| min_fp_accuracy_guard   | recall            |             -0.064054 |     -0.077449 |     -0.051341 | False           |
| min_fp_accuracy_guard   | f1                |             -0.009702 |     -0.021607 |      0.001687 | True            |
| min_fp_accuracy_guard   | f0_5              |              0.027269 |      0.015170 |      0.039182 | False           |
| min_fp_accuracy_guard   | balanced_accuracy |             -0.011377 |     -0.018673 |     -0.004318 | False           |
| max_precision_recall_50 | FP                |           -141.000000 |   -164.000000 |   -119.000000 | False           |
| max_precision_recall_50 | TP                |            -53.000000 |    -67.000000 |    -39.000000 | False           |
| max_precision_recall_50 | FN                |             53.000000 |     39.000000 |     67.000000 | False           |
| max_precision_recall_50 | accuracy          |              0.014667 |      0.010333 |      0.019333 | False           |
| max_precision_recall_50 | precision         |              0.040449 |      0.029666 |      0.051591 | False           |
| max_precision_recall_50 | recall            |             -0.039940 |     -0.050769 |     -0.029629 | False           |
| max_precision_recall_50 | f1                |             -0.001498 |     -0.011050 |      0.007688 | True            |
| max_precision_recall_50 | f0_5              |              0.023203 |      0.013459 |      0.033134 | False           |
| max_precision_recall_50 | balanced_accuracy |             -0.004883 |     -0.010882 |      0.000830 | True            |
| f0_5_optimized          | FP                |           -317.000000 |   -352.000000 |   -283.000000 | False           |
| f0_5_optimized          | TP                |           -164.000000 |   -189.000000 |   -140.000000 | False           |
| f0_5_optimized          | FN                |            164.000000 |    140.000000 |    189.000000 | False           |
| f0_5_optimized          | accuracy          |              0.025500 |      0.018333 |      0.032667 | False           |
| f0_5_optimized          | precision         |              0.096007 |      0.076789 |      0.115670 | False           |
| f0_5_optimized          | recall            |             -0.123587 |     -0.141430 |     -0.106012 | False           |
| f0_5_optimized          | f1                |             -0.033597 |     -0.050267 |     -0.017028 | False           |
| f0_5_optimized          | f0_5              |              0.037698 |      0.020417 |      0.054749 | False           |
| f0_5_optimized          | balanced_accuracy |             -0.027875 |     -0.037415 |     -0.018299 | False           |

## 10. Manual versus sklearn parity

All 6000 legacy rows were scored independently by `CustomV2KNNClassifier` and scikit-learn. Maximum absolute probability difference was 0.00908; 8 rows differed by more than 1e-10.

| operating_point         |   agreement |   disagreements |
|:------------------------|------------:|----------------:|
| current_v2_threshold    |        6000 |               0 |
| min_fp_recall_50        |        6000 |               0 |
| min_fp_f1_52            |        6000 |               0 |
| min_fp_accuracy_guard   |        6000 |               0 |
| max_precision_recall_50 |        6000 |               0 |
| f0_5_optimized          |        5999 |               1 |

## 11. Limitations

The legacy test has prior analyst exposure and provides comparability rather than independent validation. Bootstrap intervals characterize this fixed legacy sample. Threshold constraints encode technical trade-offs rather than monetary costs. Results apply to the frozen V2 scores and observed class prevalence. No new independent data was collected.

## 12. Conclusion

False positives can be reduced materially under the predeclared OOF constraints, but the reduction costs true positives. `f0_5_optimized` produced the largest observed reduction: 317 fewer FP, recall change -0.123587, F1 change -0.033597, and 51.735 TP lost per 100 FP avoided. Among the recall/F1/accuracy floor constraints, `min_fp_f1_52` removed the most FP. The accuracy-guard point stayed within its declared OOF guard by construction and changed legacy accuracy by +0.018000; its constraint was not reopened. The F0.5 point is the most conservative toward false-positive avoidance. The coincident recall-constrained and precision-first point is the most balanced observed lower-FP trade-off because it retained more sensitivity (legacy precision 0.559048, recall 0.495855) and changed F1 by only -0.001498. The accuracy-guard point remains the explicit multi-floor alternative. These descriptions name their objectives; none is universally best.
