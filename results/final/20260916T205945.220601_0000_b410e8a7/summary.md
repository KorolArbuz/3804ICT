# Final comparison

Run: `20260916T205945.220601_0000_b410e8a7`; manifest: [run_manifest.json](run_manifest.json).

These are fixed-data software reproduction results on a previously inspected legacy test. They are not new independent statistical validation.

The V5.1 baseline uses a different model and workload (k=19, uniform vote, original features). The four final implementations use k=101, inverse-distance voting and the frozen balanced_low_fp threshold. Weka retains native IBk distribution semantics.

| implementation_id | f1 | recall | precision | average_precision | roc_auc | accuracy | balanced_accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_python_v5_1 | 0.439120 | 0.338357 | 0.625348 | 0.485337 | 0.735293 | 0.808833 | 0.640396 | 4404 | 269 | 878 | 449 |
| final_python | 0.525559 | 0.495855 | 0.559048 | 0.532645 | 0.765655 | 0.802000 | 0.692396 | 4154 | 519 | 669 | 658 |
| final_sklearn | 0.525559 | 0.495855 | 0.559048 | 0.532630 | 0.765646 | 0.802000 | 0.692396 | 4154 | 519 | 669 | 658 |
| final_cpp | 0.525559 | 0.495855 | 0.559048 | 0.532645 | 0.765655 | 0.802000 | 0.692396 | 4154 | 519 | 669 | 658 |
| final_weka | 0.525769 | 0.495855 | 0.559524 | 0.540097 | 0.767100 | 0.802167 | 0.692503 | 4155 | 518 | 669 | 658 |

## Final implementation agreement

| implementation_a | implementation_b | rows | label_disagreements | label_agreement | max_abs_score_difference | mean_abs_score_difference | score_tolerance | scores_over_tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_python | final_sklearn | 6000 | 0 | 1.0 | 0.009083497248437489 | 1.048392099521347e-05 | 1e-12 | 10 |
| final_python | final_cpp | 6000 | 0 | 1.0 | 7.771561172376096e-16 | 5.949667841731454e-17 | 1e-12 | 0 |
| final_python | final_weka | 6000 | 1 | 0.9998333333333334 | 0.32717972755992286 | 0.0005301987909644525 | 1e-12 | 6000 |
| final_sklearn | final_cpp | 6000 | 0 | 1.0 | 0.009083497248437433 | 1.0483920995226374e-05 | 1e-12 | 10 |
| final_sklearn | final_weka | 6000 | 1 | 0.9998333333333334 | 0.32717972755992286 | 0.0005312487962924407 | 1e-12 | 6000 |
| final_cpp | final_weka | 6000 | 1 | 0.9998333333333334 | 0.32717972755992286 | 0.0005301987909644545 | 1e-12 | 6000 |

The training-OOF recall >= 0.50 constraint is a selection criterion; it does not guarantee test recall.

## Runtime on the common final workload

Prediction phase only; warmups excluded. Fit, startup, loading and end-to-end scopes remain separately recorded in benchmark CSVs.

| implementation_id | n | median_seconds | min_seconds | max_seconds | q25_seconds | q75_seconds | iqr_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_cpp | 20 | 3.6774041 | 3.6739259 | 3.7111711 | 3.67662835 | 3.683515675 | 0.006887325000000111 |
| final_python | 20 | 35.661570199998096 | 35.167595400009304 | 36.34434189996682 | 35.447547650008346 | 35.79999705000955 | 0.3524494000012055 |
| final_sklearn | 20 | 1.4031036499945913 | 1.3998691000160761 | 1.4073109999881126 | 1.4020096750027733 | 1.4038521999900695 | 0.0018425249872962013 |
| final_weka | 20 | 13.35927495 | 13.314506 | 13.4557347 | 13.3388958 | 13.3867821 | 0.04788630000000005 |

## Baseline timing: different model and workload

| implementation_id | n | median_seconds | min_seconds | max_seconds | q25_seconds | q75_seconds | iqr_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_python_v5_1 | 20 | 1.2562761499721091 | 1.2499163000029512 | 1.2921021999791265 | 1.254726374972961 | 1.25978502498765 | 0.005058650014689192 |

A baseline/final timing ratio is not a same-model implementation speedup.

