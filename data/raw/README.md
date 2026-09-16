# Credit-default dataset

Download the official **Default of Credit Card Clients** dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
([dataset DOI](https://doi.org/10.24432/C55S3H)). The official spreadsheet is commonly
named `default of credit card clients.xls`. Place the downloaded file in this directory
or pass its path directly:

```text
python run_all.py --data "data/raw/default of credit card clients.xls" --build
```

The current recorded experiment uses `data/raw/UCI_Credit_Card.csv`, a CSV representation
of the same 30,000-row dataset, with the original ID and target retained. If you use CSV,
preserve source row order and values, and pass its exact filename. CSV, XLS and XLSX
loaders normalize the known UCI column names. The target must contain both binary
classes; original IDs must not be used as predictors.

Each experiment records the supplied file's raw SHA-256 and the canonical split/matrix
hashes. A different file encoding can change the file hash without changing numeric
data; a different row order can change the fixed split and invalidates exact reproduction.
Do not silently substitute another credit dataset or reorder rows to match results.

Raw data is not included in the default source submission ZIP or committed by the
project. The clean-install verifier receives an external dataset path. No public
dataset commit is required to run the project.
