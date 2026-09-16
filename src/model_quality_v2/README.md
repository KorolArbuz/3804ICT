# Model Quality V2

This package runs a staged 5-outer × 3-inner nested cross-validation study entirely inside the original 24,000-row training partition. It examines k, money transforms, PAY-code representation, two small engineered feature blocks, and PAY/delinquency group weighting. Each stage compares one hypothesis family with the retained prior configuration.

All continuous PAY/delinquency components and fold-learned neutral category indicators are standardized on the relevant training fold before the optional group multiplier is applied.

Run the complete study and the post-freeze legacy comparison:

```bash
python -m src.model_quality_v2 --data data/raw/UCI_Credit_Card.csv
```

Stop after a stage or resume completed work:

```bash
python -m src.model_quality_v2 --data data/raw/UCI_Credit_Card.csv --stage 2
python -m src.model_quality_v2 --data data/raw/UCI_Credit_Card.csv --resume
```

Completed stages are not silently overwritten. `--force-stage N` explicitly recomputes stage N and all later stages. The final configuration is frozen before the existing 6,000-row test is scored. That test is reported only for historical continuity because it was examined in earlier work and is no longer independent validation.
