"""Fit both representations on the same training rows and export canonical inputs."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.common.experiment import array_hash, sha256
from src.common.utils import write_json
from src.data.dataset import find_dataset, load_dataset, split_dataset
from .pipeline import make_preprocessor
from .transforms import V2Preprocessor


def export_group(
    directory, training, testing, y_train, y_test, train_ids, test_ids, originals, names
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, X, y, ids in [
        ("train", training, y_train, train_ids),
        ("test", testing, y_test, test_ids),
    ]:
        np.savez_compressed(
            directory / f"{name}.npz",
            X=X,
            y=y,
            row_ids=ids,
            original_ids=originals[ids],
            feature_names=names,
        )
        np.savetxt(directory / f"{name}_features.csv", X, delimiter=",", fmt="%.17g")
        # Query labels never enter either external prediction engine.
        csv_lines = (
            (directory / f"{name}_features.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        header = (
            ["@RELATION credit_default", ""]
            + [f"@ATTRIBUTE '{feature}' NUMERIC" for feature in names]
            + ["@ATTRIBUTE class {0,1}", "", "@DATA"]
        )
        rows = [
            line + "," + (str(int(y[i])) if name == "train" else "?")
            for i, line in enumerate(csv_lines)
        ]
        (directory / f"{name}.arff").write_text(
            "\n".join(header + rows) + "\n", encoding="utf-8"
        )
        decoded = np.loadtxt(directory / f"{name}_features.csv", delimiter=",", ndmin=2)
        if not np.array_equal(decoded, X):
            raise ValueError("Canonical CSV round trip changed numeric values")
        arff_values = np.asarray(
            [[float(v) for v in row.rsplit(",", 1)[0].split(",")] for row in rows]
        )
        if not np.array_equal(arff_values, X):
            raise ValueError("Canonical ARFF round trip changed numeric values")
        with np.load(directory / f"{name}.npz", allow_pickle=False) as loaded:
            if not np.array_equal(loaded["X"], X) or not np.array_equal(
                loaded["row_ids"], ids
            ):
                raise ValueError("NPZ round trip changed data/order")
    np.savetxt(directory / "train_labels.csv", y_train, fmt="%d")
    return {
        path.name: sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def prepare(data, directory, configs):
    source = find_dataset(data)
    dataset = load_dataset(source)
    train, test = split_dataset(dataset)
    # Windows NumPy defaults source positions to int32; the portable ID contract is int64.
    train = train.astype(np.int64, copy=False)
    test = test.astype(np.int64, copy=False)
    if set(train) & set(test):
        raise ValueError("Train/test overlap")
    result = {}
    manifest = {
        "schema_version": 1,
        "dataset": {
            "filename": source.name,
            "sha256": sha256(source),
            "rows": len(dataset.y),
        },
        "split": {
            "seed": 42,
            "test_fraction": 0.2,
            "stratified": True,
            "sorted_source_order": True,
            "train_row_ids_hash": array_hash(train),
            "test_row_ids_hash": array_hash(test),
            "y_test_hash": array_hash(dataset.y[test]),
            "original_test_ids_hash": array_hash(dataset.original_ids[test]),
        },
        "numeric_hash_definition": "SHA256(sorted compact JSON dtype/shape/order + LF + C-contiguous little-endian bytes); strings use semantic JSON hash",
        "groups": {},
    }
    for group in ("baseline", "final"):
        preprocessor = (
            make_preprocessor(dataset.X)
            if group == "baseline"
            else V2Preprocessor(SimpleNamespace(**configs[group]["model"]))
        )
        X_train = np.ascontiguousarray(
            preprocessor.fit_transform(dataset.X.iloc[train], dataset.y[train]),
            dtype=np.float64,
        )
        X_test = np.ascontiguousarray(
            preprocessor.transform(dataset.X.iloc[test]), dtype=np.float64
        )
        names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise ValueError("Nonfinite prepared data")
        group_dir = Path(directory) / group
        files = export_group(
            group_dir,
            X_train,
            X_test,
            dataset.y[train],
            dataset.y[test],
            train,
            test,
            dataset.original_ids,
            names,
        )
        result[group] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": dataset.y[train],
            "y_test": dataset.y[test],
            "train_ids": train,
            "row_ids": test,
            "original_ids": dataset.original_ids[test],
            "names": names,
            "directory": group_dir,
        }
        manifest["groups"][group] = {
            "config_hash": configs[group]["config_hash"],
            "training_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "dtype": "<f8",
            "feature_names": names.tolist(),
            "train_numeric_hash": array_hash(X_train),
            "test_numeric_hash": array_hash(X_test),
            "files": files,
            "round_trip": "PASS: NPZ/CSV/ARFF exact values and ordering",
            "weka_query_classes": "all missing",
        }
    write_json(Path(directory).parent / "data_manifest.json", manifest)
    return result, manifest
