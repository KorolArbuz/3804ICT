from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import (
    CATEGORICAL_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_DIR,
    TEST_SIZE,
)

TARGET_NAMES = {"DEFAULT_PAYMENT_NEXT_MONTH", "Y", "TARGET"}
UCI_FEATURE_NAMES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
]


def normalize_column(name):
    return re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_")


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    row_ids: np.ndarray
    original_ids: np.ndarray
    summary: dict


def find_dataset(path=None):
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {candidate}")
        return candidate
    supported_extensions = {".xls", ".xlsx", ".csv"}
    candidates = [
        candidate
        for candidate in RAW_DATA_DIR.glob("*")
        if candidate.suffix.lower() in supported_extensions
    ]
    candidates.sort()

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one XLS/XLSX/CSV in {RAW_DATA_DIR}; found {len(candidates)}. "
            "Place the UCI dataset there or use --data <path>."
        )
    return candidates[0]


def build_dataset_summary(source, frame, X, y, original_ids, original_names, header_row):
    distribution = {
        str(label): int((y == label).sum())
        for label in (0, 1)
    }
    class_percentages = {
        label: 100 * count / len(frame)
        for label, count in distribution.items()
    }
    observed_categories = {
        column: sorted(X[column].dropna().unique().tolist())
        for column in CATEGORICAL_COLUMNS
        if column in X
    }

    return {
        "source_filename": source.name,
        "source_path": str(source.resolve()),
        "header_row_zero_based": header_row,
        "row_count": len(frame),
        "original_predictor_count": X.shape[1],
        "target_distribution": distribution,
        "class_percentages": class_percentages,
        "duplicate_row_count": int(frame.duplicated().sum()),
        "duplicate_predictor_target_count": int(
            frame.drop(columns="ID", errors="ignore").duplicated().sum()
        ),
        "missing_value_count": int(frame.isna().sum().sum()),
        "predictor_missing_value_count": int(X.isna().sum().sum()),
        "id_column": "ID" if "ID" in frame else None,
        "duplicate_original_id_count": int(pd.Series(original_ids).duplicated().sum()),
        "row_id_definition": "zero-based source data row position; original ID retained separately",
        "column_mapping": dict(zip(original_names, frame.columns)),
        "observed_nominal_categories": observed_categories,
        "category_policy": "Preserve all supplied numeric codes, including undocumented categories",
    }


def load_dataset(path=None):
    source = find_dataset(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        header_row = 0
    elif source.suffix.lower() in {".xls", ".xlsx"}:
        if source.suffix.lower() == ".xls":
            engine = "xlrd"
        else:
            engine = "openpyxl"

        preview = pd.read_excel(source, header=None, nrows=12, engine=engine)
        header_row = None
        for row_number, row in preview.iterrows():
            normalized_values = {normalize_column(value) for value in row}
            has_target = normalized_values.intersection(TARGET_NAMES)
            has_known_column = normalized_values.intersection({"SEX", "LIMIT_BAL", "ID"})
            if has_target and has_known_column:
                header_row = int(row_number)
                break

        if header_row is None:
            raise ValueError("Cannot locate Excel header in the first 12 rows")
        frame = pd.read_excel(source, header=header_row, engine=engine)
    else:
        raise ValueError("Supported formats are CSV, XLS and XLSX")

    original_names = [str(c) for c in frame.columns]
    names = [normalize_column(c) for c in frame.columns]
    if {f"X{i}" for i in range(1, 24)}.issubset(names):
        aliases = {f"X{i}": name for i, name in enumerate(UCI_FEATURE_NAMES, start=1)}
        names = [aliases.get(name, name) for name in names]
    if len(set(names)) != len(names):
        raise ValueError("Column names collide after normalization")

    frame.columns = names
    targets = [c for c in names if c in TARGET_NAMES]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target column; found {targets}")

    frame = frame.rename(columns={targets[0]: "target"})
    y_numeric = pd.to_numeric(frame["target"], errors="raise")
    if y_numeric.isna().any() or not y_numeric.isin([0, 1]).all():
        raise ValueError("All target values must be nonmissing binary 0 or 1")
    if y_numeric.nunique() != 2:
        raise ValueError("The experiment requires both target classes")

    y = y_numeric.to_numpy(dtype=np.int64)
    if "ID" in frame:
        original_ids = frame["ID"].astype("string").fillna("<missing>").to_numpy(dtype=str)
    else:
        original_ids = np.arange(len(frame)).astype(str)

    X = frame.drop(columns=["target", "ID"], errors="ignore").copy()
    if X.shape[1] == 0:
        raise ValueError("No predictive features remain")

    for column in X:
        try:
            X[column] = pd.to_numeric(X[column], errors="raise").astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Predictor {column} contains nonnumeric codes/values") from error
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Infinite predictor values are invalid")

    summary = build_dataset_summary(source, frame, X, y, original_ids, original_names, header_row)
    row_ids = np.arange(len(frame), dtype=np.int64)
    return Dataset(X, y, row_ids, original_ids, summary)


def split_dataset(dataset):
    row_positions = np.arange(len(dataset.y))
    train_indices, test_indices = train_test_split(
        row_positions,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset.y,
    )
    return np.sort(train_indices), np.sort(test_indices)
