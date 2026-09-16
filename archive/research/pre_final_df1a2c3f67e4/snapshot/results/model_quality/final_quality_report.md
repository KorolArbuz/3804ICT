# Final model-quality analysis

## Frozen selection and test results

Training-only cross-validation selected **k=25, euclidean,
distance voting** before this test evaluation. The untouched test set was
then evaluated once with the accepted baseline and the frozen candidate.

| Metric | Baseline Custom V5.1 | Enhanced Custom | Delta |
|---|---:|---:|---:|
| Class-1 F1 | 0.4391 | 0.4447 | +0.0055 |
| Recall | 0.3384 | 0.3406 | +0.0023 |
| Precision | 0.6253 | 0.6402 | +0.0149 |
| Balanced accuracy | 0.6404 | 0.6431 | +0.0027 |
| Average precision | 0.4853 | 0.5030 | +0.0177 |
| ROC-AUC | 0.7353 | 0.7395 | +0.0042 |

The confusion counts changed by **TP +3**, **FN -3**,
and **FP -15**. This states the measured trade-off directly; accuracy
alone is not treated as evidence of improved default detection.

The arithmetic majority-class reference accuracy is **0.7788**.
This is an arithmetic majority-class reference, not a trained model.

## Paired bootstrap description

The deterministic paired bootstrap resampled the same test rows 10,000 times. Its
95% percentile intervals for candidate-minus-baseline differences were:

| Metric | Observed delta | 95% percentile interval |
|---|---:|---:|
| F1 | +0.0055 | [-0.0054, +0.0168] |
| Recall | +0.0023 | [-0.0078, +0.0127] |
| Precision | +0.0149 | [-0.0005, +0.0308] |
| Balanced accuracy | +0.0027 | [-0.0026, +0.0083] |

These intervals describe paired resampling variation; they are not automatically
labelled as statistical significance tests.

## Implementation checks and limitations

Enhanced Custom and scikit-learn differed on **0**
test labels; their maximum absolute class-1 probability difference was
**0.0287**, with
**7** rows above 1e-10. Those
rows are itemized in the probability diagnostics and reflect exact-zero,
floating-point distance, or deterministic boundary-tie handling. Weka status:
**evaluated with genuine Weka IBk**. Genuine Weka IBk uses its native inverse-distance,
distance-tie, and nominal-distribution behavior, so exact probability parity is not
assumed.

The CV F1 improvement was small, only five folds and one fixed split were used, and
the paired test bootstrap does not replace external validation. The optional
threshold study was not run; the primary experiment kept the 0.5/argmax decision
rule separate from any future threshold investigation.

The measured answer to the research question is a **small improvement** in this
fixed experiment: class-1 F1, recall, precision, balanced accuracy, average
precision, and ROC-AUC all increased after training-only selection. The recall
gain was only +0.0023, and every reported paired interval includes
zero, so the evidence supports a cautious result rather than a broad claim.
