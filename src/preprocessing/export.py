import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DATA_DIR
from src.common.utils import load_pair, read_json, write_json
from src.data.dataset import load_dataset, split_dataset
from .pipeline import make_preprocessor, validate_processed_feature_names


def export_processed(dataset, train, test, X_train, X_test, feature_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        ("train", train, X_train),
        ("test", test, X_test),
    )
    for name, row_indices, values in datasets:
        output_path = output_dir / f"{name}.npz"
        np.savez_compressed(
            output_path,
            X=values,
            y=dataset.y[row_indices],
            row_ids=dataset.row_ids[row_indices],
            original_ids=dataset.original_ids[row_indices],
            feature_names=np.array(feature_names),
        )


def preprocess(data=None, output_dir=PROCESSED_DATA_DIR, selected_path=None):
    if selected_path is not None:
        selected_parameters = read_json(selected_path)
        if int(selected_parameters["selected_k"]) < 1:
            raise ValueError("Selected k must be at least 1")

    dataset = load_dataset(data)
    train_indices, test_indices = split_dataset(dataset)
    transformer = make_preprocessor(dataset.X)

    # Fit only on training rows so information from the test set cannot leak in.
    X_train = transformer.fit_transform(dataset.X.iloc[train_indices])
    X_test = transformer.transform(dataset.X.iloc[test_indices])
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Preprocessing produced nonfinite values")

    feature_names = validate_processed_feature_names(transformer.get_feature_names_out().tolist())
    export_processed(
        dataset,
        train_indices,
        test_indices,
        X_train,
        X_test,
        feature_names,
        output_dir,
    )

    print(
        f"Preprocessed {len(train_indices)} train / {len(test_indices)} test rows; "
        f"{len(feature_names)} features -> {output_dir}",
        flush=True,
    )
    return {
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "feature_names": feature_names,
    }


def arff_text(X, y, names):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or X.shape[1] != len(names):
        raise ValueError("ARFF arrays and feature names have incompatible dimensions")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("ARFF requires finite predictors and binary targets")

    validate_processed_feature_names(names)
    lines = ["@RELATION credit_default", ""]
    for name in names:
        escaped_name = name.replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"@ATTRIBUTE '{escaped_name}' NUMERIC")

    lines.extend(["@ATTRIBUTE class {0,1}", "", "@DATA"])
    for row, label in zip(X, y):
        values = [format(float(value), ".17g") for value in row]
        values.append(str(int(label)))
        lines.append(",".join(values))

    return "\n".join(lines) + "\n"


def export_arff(train, test, output_dir):
    training, testing = load_pair(train, test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = []
    for name, dataset in (("train", training), ("test", testing)):
        feature_names = dataset["feature_names"].tolist()
        contents = arff_text(dataset["X"], dataset["y"], feature_names)
        output_path = output_dir / f"{name}.arff"
        output_path.write_text(contents, encoding="utf-8", newline="\n")
        headers.append(contents.split("@DATA")[0])

    if headers[0] != headers[1]:
        raise ValueError("ARFF train/test headers differ")

    test_ids = pd.DataFrame({
        "row_id": testing["row_ids"],
        "original_id": testing["original_ids"],
    })
    test_ids.to_csv(output_dir / "test_ids.csv", index=False)

    print(f"Exported matching ARFF headers and test row IDs -> {output_dir}")
    return {"train_rows": len(training["y"]), "test_rows": len(testing["y"])}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_cpp_features(path, values):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for row in np.asarray(values, dtype=np.float64):
            handle.write(",".join(format(float(value), ".17g") for value in row))
            handle.write("\n")


def _write_cpp_labels(path, labels):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for label in np.asarray(labels):
            handle.write(f"{int(label)}\n")


def export_cpp_csv(train, test, output_dir):
    """Export the canonical NPZ arrays without changing rows or preprocessing."""
    training, testing = load_pair(train, test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = training["feature_names"].tolist()
    validate_processed_feature_names(feature_names)
    if training["X"].shape[1] != testing["X"].shape[1]:
        raise ValueError("C++ train/test feature counts differ")

    files = {
        "train_features": output_dir / "train_features.csv",
        "train_labels": output_dir / "train_labels.csv",
        "test_features": output_dir / "test_features.csv",
        "test_labels": output_dir / "test_labels.csv",
    }
    _write_cpp_features(files["train_features"], training["X"])
    _write_cpp_labels(files["train_labels"], training["y"])
    _write_cpp_features(files["test_features"], testing["X"])
    _write_cpp_labels(files["test_labels"], testing["y"])

    manifest = {
        "format": "headerless comma-separated float64 values; one binary label per line",
        "float_format": ".17g round-trippable decimal",
        "train_rows": len(training["y"]),
        "test_rows": len(testing["y"]),
        "feature_count": training["X"].shape[1],
        "feature_names": feature_names,
        "excluded_fields": ["ID", "target", "class"],
        "source_npz_sha256": {
            "train": _sha256(train),
            "test": _sha256(test),
        },
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in files.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"Exported canonical C++ inputs: {manifest['train_rows']} train / "
        f"{manifest['test_rows']} test rows, {manifest['feature_count']} features "
        f"-> {output_dir}",
        flush=True,
    )
    return manifest
