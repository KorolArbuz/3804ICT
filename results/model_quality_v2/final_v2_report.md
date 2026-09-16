# Model Quality V2

## 1. Research question

Can KNN improve underlying score ranking and neighbour quality through feature representation, k selection, and neighbourhood geometry? Average Precision (AP) is the primary metric, followed by ROC-AUC. Threshold-selected classification metrics describe the operating point and are not evidence that ranking itself improved.

## 2. Why nested CV

All selection used five outer folds and three inner folds inside the fixed 24,000-row training partition (shuffle seed 42). Inner OOF predictions selected both configuration and threshold. Each transform was fitted on its inner-training rows. The outer labels were used once for evaluation and never entered configuration or threshold selection. The same outer folds were reused across the staged investigation, which makes the stage comparisons paired but also adaptive.

## 3. Current reference model

The starting model is k=25, Euclidean distance, inverse-distance voting, and the existing preprocessing. The earlier threshold study selected 0.279326488; it improved the classification operating point, while its AP and ROC-AUC remained score-ranking properties of the unchanged model.

## 4. Stage 1 k+threshold

The candidates were k=13, 19, 25, 31, 35, 51, 75, and 101. Each was ranked by mean inner-fold AP, then mean ROC-AUC, OOF threshold-selected F1, balanced accuracy, and smaller k. The retained configuration after this stage was `k101__money-standard__pay-raw__blocks-none__payw-1`; the stage was **accepted**.

## 5. Stage 2 money transforms

Standard scaling, signed-log plus standard scaling, Yeo-Johnson, and robust scaling were compared only for limit, bill, and payment amounts. The retained configuration was `k101__money-signed_log__pay-raw__blocks-none__payw-1`; the stage was **accepted**.

## 6. Stage 3 PAY representation

Raw scaled PAY codes were compared with fold-learned positive-delay values and neutral indicators for observed non-positive codes. Unseen categories produce zero for every learned category indicator. The retained configuration was `k101__money-signed_log__pay-structured__blocks-none__payw-1`; the stage was **accepted**.

## 7. Stage 4 engineered features

The ablation compared no block, limit-relative Block A, delinquency-summary Block B, and A+B. Ratios use missing values for missing or non-positive denominators; fold-fitted median imputation handles them. These are descriptive features, not causal variables. The retained configuration was `k101__money-signed_log__pay-structured__blocks-AB__payw-1`; the stage was **accepted**.

## 8. Stage 5 group weighting

PAY/delinquency squared-distance weights 0.5, 1.0, and 2.0 were implemented by multiplying standardized group columns by the square root of the weight. The retained configuration was `k101__money-signed_log__pay-structured__blocks-AB__payw-1`; the stage was **rejected**.

|   stage | hypothesis         | decision          |       AP |   ROC-AUC |       F1 |   AP wins |
|--------:|:-------------------|:------------------|---------:|----------:|---------:|----------:|
|       1 | k_threshold        | accepted          | 0.518556 |  0.753941 | 0.522277 |         5 |
|       2 | money_transform    | accepted          | 0.522402 |  0.768583 | 0.537759 |         3 |
|       3 | pay_representation | accepted          | 0.529105 |  0.770583 | 0.535975 |         5 |
|       4 | engineered_blocks  | accepted          | 0.539185 |  0.775682 | 0.539166 |         5 |
|       5 | group_weight       | retained baseline | 0.543313 |  0.774631 | 0.538382 |         5 |

## 9. Final configuration

The configuration was frozen before any V2 legacy-test scoring: k=101, Euclidean distance, distance voting, `signed_log` money transform, `structured` PAY representation, feature blocks `['A', 'B']`, PAY/delinquency weight 1.0, and nested-derived threshold 0.280150124. The acceptance rule required AP improvement of at least 0.002 or AP wins in at least four folds, with mean ROC-AUC decrease no worse than 0.001 and F1 decrease no worse than 0.002.

## 10. Manual V2 parity

The clarity-first manual kernel uses exact brute-force Euclidean neighbours, inverse-distance voting, exact zero-distance semantics, and original-index boundary tie handling. Against scikit-learn on 48 transformed rows, the maximum absolute probability difference was 4.17e-13 (tolerance 1e-12); parity **passed**.

## 11. Nested-CV evidence

Final-stage outer-fold values:

|   outer_fold |   average_precision |   roc_auc |       f1 |   recall |   precision |   balanced_accuracy |   threshold |
|-------------:|--------------------:|----------:|---------:|---------:|------------:|--------------------:|------------:|
|     1.000000 |            0.547004 |  0.777686 | 0.546882 | 0.557439 |    0.536718 |            0.710367 |    0.275040 |
|     2.000000 |            0.545207 |  0.777551 | 0.549929 | 0.547081 |    0.552807 |            0.710673 |    0.300631 |
|     3.000000 |            0.563606 |  0.783228 | 0.546171 | 0.587571 |    0.510221 |            0.713662 |    0.253007 |
|     4.000000 |            0.517389 |  0.766697 | 0.520561 | 0.524482 |    0.516698 |            0.692551 |    0.295915 |
|     5.000000 |            0.543359 |  0.767992 | 0.528369 | 0.561734 |    0.498745 |            0.700765 |    0.242159 |

Relative to the Stage-1 starting baseline, the final-stage adaptive procedure changed mean AP by +0.033979 and mean ROC-AUC by +0.027711. Raw paired deltas and fold win counts for every stage are in `stage_decisions.csv`.

## 12. Legacy test comparison

**Legacy held-out test comparison; test was already examined in earlier project stages and is not independent validation for V2.** No selection was reopened after these values were inspected.

| comparison                 |   average_precision |   roc_auc |       f1 |   recall |   precision |   balanced_accuracy |   TN |   FP |   FN |   TP |
|:---------------------------|--------------------:|----------:|---------:|---------:|------------:|--------------------:|-----:|-----:|-----:|-----:|
| accepted_v5_1              |            0.485337 |  0.735293 | 0.439120 | 0.338357 |    0.625348 |            0.640396 | 4404 |  269 |  878 |  449 |
| k25_distance_threshold_0.5 |            0.503025 |  0.739541 | 0.444663 | 0.340618 |    0.640227 |            0.643132 | 4419 |  254 |  875 |  452 |
| current_f1_threshold       |            0.503025 |  0.739541 | 0.513845 | 0.552374 |    0.480341 |            0.691338 | 3880 |  793 |  594 |  733 |
| final_v2_nested_threshold  |            0.532630 |  0.765646 | 0.527057 | 0.535795 |    0.518600 |            0.697279 | 4013 |  660 |  616 |  711 |

## 13. Limitations

Five outer folds give only five paired observations, so the deltas are descriptive engineering evidence rather than significance claims. Stages reuse the same folds and later hypotheses depend on earlier decisions. Only the predeclared, small candidate families were tested. The legacy test has prior analyst exposure. PAY code indicators use neutral labels because undocumented code meanings were not assumed.

## 14. Conclusion

The staged final procedure showed higher outer-fold score ranking than the starting k=25 representation. The observed mean changes were AP +0.033979 and ROC-AUC +0.027711. Threshold selection separately changed the class-1 operating point; it did not itself improve AP or ROC-AUC.
