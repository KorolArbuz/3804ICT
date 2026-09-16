# KNN model-quality experiment

This package is separate from the accepted k=19 Euclidean/uniform experiment.
It asks whether a small, predeclared KNN family improves class-1/default F1.
The fixed 80/20 split, seed 42, fold-local preprocessing, and five stratified
training folds remain unchanged. Test rows are not used for selection.

Run the training-only 64-configuration search first:

```bash
python -m src.model_quality.search --data data/raw/UCI_Credit_Card.csv
```

This writes `results/model_quality/selected_quality_parameters.json` with
`test_evaluated: false`. After reviewing that frozen training-CV artifact, run
the one-time final evaluation:

```bash
python -m src.model_quality.evaluate --data data/raw/UCI_Credit_Card.csv
```

After a completed evaluation, `--verify-rerun` may regenerate artifacts from
the same frozen parameters. It accepts only `test_evaluated: true` and never
reruns model selection.

The selected manual implementation is `EnhancedCustomKNNClassifier`. It uses
exact brute-force Euclidean neighbours and scikit-compatible inverse-distance
voting: when one or more selected neighbours have zero distance, all nonzero
neighbours receive zero effective weight. Ties at the k boundary are resolved
by original training order.

The optional threshold study is deliberately disabled. It can be investigated
later with training out-of-fold probabilities without changing this primary
k/metric/voting experiment.
