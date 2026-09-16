# Training-only model-quality report

## Measured results

Training-only cross-validation selected **k=25, euclidean,
distance voting**. The accepted k=19 Euclidean/uniform baseline had mean
class-1 F1 **0.4464**; the selected configuration had
**0.4471** (delta **+0.0007**).

| Metric | Baseline mean | Selected mean | Delta |
|---|---:|---:|---:|
| Class-1 F1 | 0.4464 | 0.4471 | +0.0007 |
| Recall | 0.3421 | 0.3424 | +0.0004 |
| Precision | 0.6427 | 0.6443 | +0.0017 |
| Balanced accuracy | 0.6441 | 0.6444 | +0.0003 |
| Average precision | 0.4916 | 0.5093 | +0.0177 |

The selected configuration's fold F1 standard deviation was 0.0121, compared with 0.0233 for the baseline. It was more consistent across these five folds, but five observed folds do not establish statistical significance.

## Interpretation

The observed change does not primarily
show a precision-for-recall trade-off relative to the accepted baseline.
The best Manhattan configuration reached mean F1 **0.4254** (-0.0210 versus the baseline), so Manhattan did not produce the winner. The best distance-weighted configuration reached **0.4471**, while the best uniform configuration reached **0.4464**; distance weighting produced the selected maximum.

The selected k differs from the
accepted k=19 setting (25 versus 19). The six-neighbour change is
numerically clear, but the mean-F1 separation of +0.0007 is small;
these five folds do not establish that the k change is materially important.

## Limitation

All selection statements come from the fixed training partition. The test set was
not evaluated or used to choose k, distance, voting, threshold, or preprocessing.
Five CV folds are not an independent significance test, so this report does not
claim statistical significance.
