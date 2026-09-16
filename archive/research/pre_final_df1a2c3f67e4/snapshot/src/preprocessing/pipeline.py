from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import CATEGORICAL_COLUMNS


def feature_groups(frame):
    if "ID" in frame or "target" in frame:
        raise ValueError("ID and target must be excluded before preprocessing")

    nominal = [column for column in CATEGORICAL_COLUMNS if column in frame.columns]
    numeric = [column for column in frame.columns if column not in nominal]
    return numeric, nominal


def validate_processed_feature_names(names):
    lowered = {str(name).lower() for name in names}
    if len(names) != len(set(names)):
        raise ValueError("Processed feature names must be unique and exclude ID/target/class")
    if lowered.intersection({"id", "target", "class"}):
        raise ValueError("Processed feature names must be unique and exclude ID/target/class")
    return list(names)


def make_preprocessor(X):
    numeric_columns, nominal_columns = feature_groups(X)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)),
    ])

    transformer = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_columns),
            ("nominal", nominal_pipeline, nominal_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    return transformer
