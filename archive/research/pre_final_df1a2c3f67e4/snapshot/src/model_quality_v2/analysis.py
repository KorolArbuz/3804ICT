"""Small summaries used by V2 artifacts and reporting."""

import json

import numpy as np
import pandas as pd

from .nested_cv import METRICS


def stage_summary(outer_results):
    rows = []
    for (stage, role), group in outer_results.groupby(["stage", "role"], sort=True):
        row = {"stage": int(stage), "role": role, "outer_folds": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def decisions_frame(decisions):
    rows = []
    for decision in decisions:
        row = {
            "stage": decision["stage"], "accepted": decision["accepted"],
            "baseline_configuration_id": decision["baseline_configuration_id"],
            "candidate_family_winner_id": decision["candidate_family_winner_id"],
            "retained_configuration_id": decision["retained_configuration_id"],
            "next_threshold": decision["next_threshold"],
        }
        for name, value in decision["conditions"].items():
            row[f"condition_{name}"] = value
        for metric, values in decision["paired_deltas"].items():
            row[f"{metric}_mean_delta"] = values["mean_delta"]
            row[f"{metric}_median_delta"] = values["median_delta"]
            row[f"{metric}_wins"] = values["wins"]
            row[f"{metric}_raw_deltas"] = json.dumps(values["raw_deltas"])
        rows.append(row)
    return pd.DataFrame(rows)


def probability_parity(reference, manual, atol=1e-12):
    reference = np.asarray(reference, dtype=float)
    manual = np.asarray(manual, dtype=float)
    if reference.shape != manual.shape:
        raise ValueError("Parity arrays differ in shape")
    maximum = float(np.max(np.abs(reference - manual))) if reference.size else 0.0
    return {"rows": len(reference), "maximum_absolute_probability_difference": maximum,
            "tolerance": atol, "passed": bool(maximum <= atol)}
