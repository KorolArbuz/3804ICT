"""Small, predeclared engineered feature blocks for Model Quality V2."""

import numpy as np
import pandas as pd


MONEY_COLUMNS = (
    "LIMIT_BAL",
    *(f"BILL_AMT{i}" for i in range(1, 7)),
    *(f"PAY_AMT{i}" for i in range(1, 7)),
)
PAY_STATUS_COLUMNS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
NOMINAL_COLUMNS = ("SEX", "EDUCATION", "MARRIAGE")
OTHER_NUMERIC_COLUMNS = ("AGE",)
BLOCK_A_COLUMNS = (
    "ratio_bill_amt1_to_limit",
    "ratio_mean_bill_to_limit",
    "ratio_mean_payment_to_limit",
    "ratio_bill_change_to_limit",
    "limit_denominator_valid",
)
BLOCK_B_COLUMNS = (
    "delinquency_positive_count",
    "delinquency_positive_max",
    "delinquency_positive_mean",
    "delinquency_pay0_positive",
)


def signed_log(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = np.isfinite(denominator) & (denominator > 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def add_engineered_features(frame, blocks):
    values = frame.copy()
    blocks = set(blocks)
    if "A" in blocks:
        limit = values["LIMIT_BAL"].to_numpy(dtype=float)
        bills = values[[f"BILL_AMT{i}" for i in range(1, 7)]].to_numpy(dtype=float)
        payments = values[[f"PAY_AMT{i}" for i in range(1, 7)]].to_numpy(dtype=float)
        values["ratio_bill_amt1_to_limit"] = safe_divide(bills[:, 0], limit)
        values["ratio_mean_bill_to_limit"] = safe_divide(
            np.nanmean(bills, axis=1), limit
        )
        values["ratio_mean_payment_to_limit"] = safe_divide(
            np.nanmean(payments, axis=1), limit
        )
        values["ratio_bill_change_to_limit"] = safe_divide(
            bills[:, 0] - bills[:, 5], limit
        )
        values["limit_denominator_valid"] = (np.isfinite(limit) & (limit > 0)).astype(
            float
        )
    if "B" in blocks:
        pay = values[list(PAY_STATUS_COLUMNS)].to_numpy(dtype=float)
        positive = np.where(np.isnan(pay), np.nan, np.maximum(pay, 0.0))
        values["delinquency_positive_count"] = np.sum(pay > 0, axis=1).astype(float)
        with np.errstate(all="ignore"):
            values["delinquency_positive_max"] = np.nanmax(positive, axis=1)
            values["delinquency_positive_mean"] = np.nanmean(positive, axis=1)
        values["delinquency_pay0_positive"] = positive[:, 0]
    return values
