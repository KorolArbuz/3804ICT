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

## Separate decision-threshold study

The model is selected first and remains frozen at **k=25, Euclidean distance,
distance weighting**. A separate study selects thresholds from five-fold
training-only out-of-fold class-1 scores, with preprocessing fitted inside each
fold. It maximizes F1 and separately maximizes recall subject to a predeclared
OOF precision floor of 0.55. This floor does not guarantee test-set precision.

Generate OOF scores, freeze the thresholds and create the training-only report:

```bash
python -m src.model_quality.threshold --data data/raw/UCI_Credit_Card.csv --oof-only
```

The held-out partition is excluded from threshold selection. Only after the
thresholds and input hashes are frozen can the final evaluation run:

```bash
python -m src.model_quality.threshold --data data/raw/UCI_Credit_Card.csv --evaluate-test
```

Results are written separately under `results/model_quality/threshold_study/`.
Normal commands refuse to overwrite a completed study. To verify a completed
evaluation using the same frozen thresholds and hashes:

```bash
python -m src.model_quality.threshold --data data/raw/UCI_Credit_Card.csv --verify-rerun
```

The rule is `probability_class_1 >= threshold`. A threshold changes the operating
point, not the neighbours, weighting, or model configuration. ROC-AUC and average
precision are score-ranking metrics and remain unchanged when only the decision
threshold changes. The default 0.5 result remains the reference, and reports
separate model-configuration gains from threshold-only gains. Enhanced Custom
supplies the primary test scores; scikit-learn is checked for agreement. Weka's
native probability semantics are not assumed equivalent, and no Weka threshold
evaluation is included.
