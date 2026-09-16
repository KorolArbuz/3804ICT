"""Fold-fitted preprocessing variants for the staged V2 investigation."""

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from .features import (
    BLOCK_A_COLUMNS,
    BLOCK_B_COLUMNS,
    MONEY_COLUMNS,
    NOMINAL_COLUMNS,
    OTHER_NUMERIC_COLUMNS,
    PAY_STATUS_COLUMNS,
    add_engineered_features,
    signed_log,
)


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return signed_log(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _category_name(value):
    value = float(value)
    if value < 0:
        return f"negative_code_{format(abs(value), 'g').replace('.', '_')}"
    return f"nonpositive_code_{format(value, 'g').replace('.', '_')}"


class StructuredPayTransformer(BaseEstimator, TransformerMixin):
    """Positive delay plus neutral indicators learned for non-positive codes."""

    def __init__(self, columns=PAY_STATUS_COLUMNS):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError("Unexpected PAY status shape")
        self.categories_ = [
            np.sort(
                np.unique(
                    values[
                        np.isfinite(values[:, index]) & (values[:, index] <= 0), index
                    ]
                )
            )
            for index in range(values.shape[1])
        ]
        positive = np.where(np.isnan(values), np.nan, np.maximum(values, 0.0))
        self.medians_ = np.nanmedian(positive, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        raw = self._raw_features(values)
        self.means_ = raw.mean(axis=0)
        self.scales_ = raw.std(axis=0)
        self.scales_ = np.where(self.scales_ > 0, self.scales_, 1.0)
        names = [f"{column}__positive_delay" for column in self.columns]
        for column, categories in zip(self.columns, self.categories_):
            names.extend(f"{column}__{_category_name(value)}" for value in categories)
        self.feature_names_out_ = np.asarray(names, dtype=object)
        return self

    def _raw_features(self, values):
        positive = np.where(np.isnan(values), np.nan, np.maximum(values, 0.0))
        positive = np.where(np.isnan(positive), self.medians_[None, :], positive)
        parts = [positive]
        for index, categories in enumerate(self.categories_):
            for category in categories:
                parts.append((values[:, index] == category).astype(float)[:, None])
        return np.column_stack(parts)

    def transform(self, X):
        if not hasattr(self, "categories_"):
            raise ValueError("fit must be called before transform")
        values = np.asarray(X, dtype=np.float64)
        raw = self._raw_features(values)
        return (raw - self.means_[None, :]) / self.scales_[None, :]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_.copy()


def _scaled_pipeline(scaler):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", scaler),
        ]
    )


def _money_pipeline(name):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name == "standard":
        steps.append(("scaler", StandardScaler()))
    elif name == "signed_log":
        steps.extend(
            (("signed_log", SignedLogTransformer()), ("scaler", StandardScaler()))
        )
    elif name == "yeo_johnson":
        steps.append(
            ("scaler", PowerTransformer(method="yeo-johnson", standardize=True))
        )
    elif name == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        raise ValueError("Unknown money transform")
    return Pipeline(steps)


class V2Preprocessor:
    def __init__(self, configuration):
        self.configuration = configuration

    def _prepare(self, frame):
        return add_engineered_features(frame, self.configuration.feature_blocks)

    def fit(self, frame, y=None):
        prepared = self._prepare(frame)
        transformers = [
            (
                "money",
                _money_pipeline(self.configuration.money_transform),
                list(MONEY_COLUMNS),
            ),
            ("other", _scaled_pipeline(StandardScaler()), list(OTHER_NUMERIC_COLUMNS)),
        ]
        if self.configuration.pay_representation == "raw":
            transformers.append(
                ("pay", _scaled_pipeline(StandardScaler()), list(PAY_STATUS_COLUMNS))
            )
        else:
            transformers.append(
                ("pay_struct", StructuredPayTransformer(), list(PAY_STATUS_COLUMNS))
            )
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                drop=None, handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                list(NOMINAL_COLUMNS),
            )
        )
        if "A" in self.configuration.feature_blocks:
            transformers.append(
                ("block_a", _scaled_pipeline(StandardScaler()), list(BLOCK_A_COLUMNS))
            )
        if "B" in self.configuration.feature_blocks:
            transformers.append(
                ("block_b", _scaled_pipeline(StandardScaler()), list(BLOCK_B_COLUMNS))
            )
        self.transformer_ = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=True,
        )
        self.transformer_.fit(prepared, y)
        self.feature_names_out_ = np.asarray(
            self.transformer_.get_feature_names_out(), dtype=str
        )
        if len(self.feature_names_out_) != len(set(self.feature_names_out_)):
            raise ValueError("V2 processed feature names must be unique")
        lowered = {name.lower() for name in self.feature_names_out_}
        if lowered.intersection({"id", "target", "class"}):
            raise ValueError("ID/target/class cannot be a V2 predictor")
        self.pay_group_indices_ = np.flatnonzero(
            [
                name.startswith(("pay__", "pay_struct__", "block_b__"))
                for name in self.feature_names_out_
            ]
        )
        return self

    def transform(self, frame):
        if not hasattr(self, "transformer_"):
            raise ValueError("fit must be called before transform")
        values = np.asarray(
            self.transformer_.transform(self._prepare(frame)), dtype=np.float64
        )
        values[:, self.pay_group_indices_] *= math.sqrt(
            self.configuration.pay_group_weight
        )
        if not np.isfinite(values).all():
            raise ValueError("V2 preprocessing produced nonfinite values")
        return values

    def fit_transform(self, frame, y=None):
        return self.fit(frame, y).transform(frame)

    def get_feature_names_out(self):
        return self.feature_names_out_.copy()
