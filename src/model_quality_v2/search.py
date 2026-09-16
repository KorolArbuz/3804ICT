"""Orchestrate staged nested CV and the post-freeze legacy comparison."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from src.common.config import PROJECT_ROOT, RANDOM_STATE, RESULTS_DIR
from src.data.dataset import find_dataset, load_dataset, split_dataset

from .analysis import decisions_frame, probability_parity, stage_summary
from .config import STAGE_NAMES, V2Configuration, stage_candidates
from .custom_v2_knn import CustomV2KNNClassifier
from .nested_cv import deterministic_splits, run_stage
from .thresholding import THRESHOLD_RULE, score_metrics
from .transforms import V2Preprocessor

OUTPUT_DIR = RESULTS_DIR / "model_quality_v2"
PROTECTED_PATHS = (
    "src/custom_knn/classifier.py",
    "src/model_quality/enhanced_custom_knn.py",
    "src/preprocessing/pipeline.py",
    "results/model_quality/selected_quality_parameters.json",
    "results/model_quality/threshold_study/threshold_selection.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protected_hashes():
    result = {name: sha256(PROJECT_ROOT / name) for name in PROTECTED_PATHS}
    for root in (PROJECT_ROOT / "results/runtime_benchmarks", PROJECT_ROOT / "cpp"):
        if root.exists():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                result[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = sha256(path)
    return result


def _git(*arguments):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", *arguments],
        cwd=PROJECT_ROOT, text=True,
    ).strip()


def create_reference_manifest(dataset, source, train, test, output_dir):
    quality = PROJECT_ROOT / "results/model_quality"
    selected = read_json(quality / "selected_quality_parameters.json")
    threshold = read_json(quality / "threshold_study/threshold_selection.json")
    comparison = pd.read_csv(quality / "test_quality_comparison.csv").to_dict("records")
    manifest = {
        "created_before_v2_selection": True,
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_at_freeze": _git("status", "--short"),
        "dataset_path": str(source.resolve()), "dataset_sha256": sha256(source),
        "split_seed": RANDOM_STATE, "training_rows": len(train), "legacy_test_rows": len(test),
        "current_model": {"k": 25, "metric": "euclidean", "weights": "distance"},
        "current_model_selection": selected,
        "threshold_study": {
            "selection_source": threshold["selection_source"],
            "default_threshold": threshold["default_threshold"],
            "f1_optimized_threshold": threshold["f1_optimized_threshold"],
            "recall_oriented_threshold": threshold["recall_oriented_threshold"],
            "oof_score_metrics": threshold["oof_score_metrics"],
            "oof_metrics": threshold["oof_metrics"],
        },
        "historical_test_metrics": comparison,
        "historical_test_context": (
            "HISTORICAL COMPARABILITY ONLY: the 6,000-row test was examined before V2 "
            "and is not independent validation."
        ),
        "protected_hashes": protected_hashes(),
    }
    write_json(output_dir / "reference_manifest.json", manifest)
    return manifest


def _load_frames(output_dir):
    inner_path = output_dir / "nested_inner_selection.csv"
    outer_path = output_dir / "nested_outer_results.csv"
    inner = pd.read_csv(inner_path) if inner_path.exists() else pd.DataFrame()
    outer = pd.read_csv(outer_path) if outer_path.exists() else pd.DataFrame()
    return inner, outer


def _persist(output_dir, inner, outer, decisions, state):
    inner.to_csv(output_dir / "nested_inner_selection.csv", index=False, float_format="%.17g")
    outer.to_csv(output_dir / "nested_outer_results.csv", index=False, float_format="%.17g")
    stage_summary(outer).to_csv(output_dir / "stage_results.csv", index=False, float_format="%.17g")
    decisions_frame(decisions).to_csv(output_dir / "stage_decisions.csv", index=False,
                                      float_format="%.17g")
    write_json(output_dir / "stage_decisions.json", decisions)
    write_json(output_dir / "search_state.json", state)
    stage1_inner = inner[inner.stage == 1]
    stage1_outer = outer[outer.stage == 1]
    stage1_inner.to_csv(output_dir / "stage1_k_threshold_inner.csv", index=False,
                        float_format="%.17g")
    stage1_outer.to_csv(output_dir / "stage1_k_threshold_outer.csv", index=False,
                        float_format="%.17g")


def _source_hashes():
    root = PROJECT_ROOT / "src/model_quality_v2"
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted(root.glob("*.py"))}


def freeze_final_configuration(configuration, threshold, decisions, outer, output_dir):
    path = output_dir / "final_v2_configuration.json"
    if path.exists():
        raise ValueError("Final V2 configuration is already frozen")
    final_role = "candidate" if not decisions or decisions[-1]["accepted"] else "baseline"
    final_stage = int(outer.stage.max())
    final_outer = outer[(outer.stage == final_stage) & (outer.role == final_role)].sort_values("outer_fold")
    evidence = {
        metric: [float(value) for value in final_outer[metric]]
        for metric in ("average_precision", "roc_auc", "f1", "recall", "precision")
    }
    value = {
        "frozen": True, "selection_data": "original 24,000-row training partition only",
        "configuration": configuration.to_dict(),
        "preprocessing": {
            "money_transform": configuration.money_transform,
            "pay_representation": configuration.pay_representation,
            "engineered_blocks": list(configuration.feature_blocks),
            "pay_delinquency_squared_distance_weight": configuration.pay_group_weight,
            "pay_delinquency_feature_multiplier": float(np.sqrt(configuration.pay_group_weight)),
        },
        "threshold": float(threshold), "threshold_selection_rule": THRESHOLD_RULE,
        "nested_cv": {"outer_folds": 5, "inner_folds": 3, "seed": RANDOM_STATE,
                      "stage_decisions": decisions, "final_evidence_role": final_role,
                      "final_outer_fold_values": evidence},
        "source_hashes": _source_hashes(), "protected_reference_hashes": protected_hashes(),
        "legacy_test_evaluated": False,
    }
    write_json(path, value)
    return value


def _historical_legacy_rows():
    quality = PROJECT_ROOT / "results/model_quality"
    old = pd.read_csv(quality / "test_quality_comparison.csv")
    thresholds = pd.read_csv(quality / "threshold_study/test_threshold_comparison.csv")
    base = old[old.implementation == "baseline_custom_v5_1"].iloc[0].to_dict()
    selected = old[old.implementation == "enhanced_custom"].iloc[0].to_dict()
    threshold = thresholds[thresholds.threshold_name == "f1_optimized_threshold"].iloc[0].to_dict()
    fields = ("average_precision", "roc_auc", "f1", "recall", "precision",
              "balanced_accuracy", "TN", "FP", "FN", "TP")
    rows = [{"comparison": "accepted_v5_1", **{key: base[key] for key in fields}},
            {"comparison": "k25_distance_threshold_0.5", **{key: selected[key] for key in fields}}]
    rows.append({"comparison": "current_f1_threshold",
                 "average_precision": selected["average_precision"], "roc_auc": selected["roc_auc"],
                 **{key: threshold[key] for key in fields if key not in {"average_precision", "roc_auc"}}})
    return rows


def evaluate_legacy(dataset, train, test, final, output_dir):
    if output_dir.joinpath("legacy_test_comparison.csv").exists():
        raise ValueError("Legacy test was already evaluated; refusing to overwrite it")
    configuration = V2Configuration.from_dict(final["configuration"])
    preprocessor = V2Preprocessor(configuration)
    X_train = preprocessor.fit_transform(dataset.X.iloc[train], dataset.y[train])
    X_test = preprocessor.transform(dataset.X.iloc[test])
    with threadpool_limits(limits=1):
        sklearn_model = KNeighborsClassifier(
            n_neighbors=configuration.k, metric="euclidean", weights="distance",
            algorithm="brute", n_jobs=1,
        ).fit(X_train, dataset.y[train])
        scores = sklearn_model.predict_proba(X_test)[:, 1]
    final_metrics = score_metrics(dataset.y[test], scores, final["threshold"])
    rows = _historical_legacy_rows()
    rows.append({"comparison": "final_v2_nested_threshold", **final_metrics})
    frame = pd.DataFrame(rows)
    frame.insert(1, "validation_context", (
        "Legacy held-out test comparison; test was already examined in earlier project "
        "stages and is not independent validation for V2."
    ))
    frame.to_csv(output_dir / "legacy_test_comparison.csv", index=False, float_format="%.17g")

    parity_count = min(48, len(X_test))
    manual = CustomV2KNNClassifier(configuration.k).fit(X_train, dataset.y[train])
    manual_scores = manual.predict_proba(X_test[:parity_count])[:, 1]
    legacy_parity = probability_parity(scores[:parity_count], manual_scores)
    legacy_parity.update({"sample": "first 48 transformed legacy rows after final freeze",
                          "passed_predictions": bool(np.array_equal(
                              sklearn_model.predict(X_test[:parity_count]),
                              manual.predict(X_test[:parity_count]),
                          ))})

    train_frame = dataset.X.iloc[train].reset_index(drop=True)
    train_labels = dataset.y[train]
    outer_fit, outer_validation = deterministic_splits(train_labels, 5)[0]
    outer_preprocessor = V2Preprocessor(configuration)
    outer_X_fit = outer_preprocessor.fit_transform(train_frame.iloc[outer_fit], train_labels[outer_fit])
    outer_X_validation = outer_preprocessor.transform(
        train_frame.iloc[outer_validation[:parity_count]]
    )
    with threadpool_limits(limits=1):
        outer_sklearn = KNeighborsClassifier(
            n_neighbors=configuration.k, metric="euclidean", weights="distance",
            algorithm="brute", n_jobs=1,
        ).fit(outer_X_fit, train_labels[outer_fit])
        outer_reference = outer_sklearn.predict_proba(outer_X_validation)[:, 1]
    outer_manual = CustomV2KNNClassifier(configuration.k).fit(
        outer_X_fit, train_labels[outer_fit]
    )
    outer_manual_scores = outer_manual.predict_proba(outer_X_validation)[:, 1]
    outer_parity = probability_parity(outer_reference, outer_manual_scores)
    outer_parity.update({"sample": "first 48 transformed outer-fold-1 validation rows",
                         "passed_predictions": bool(np.array_equal(
                             outer_sklearn.predict(outer_X_validation),
                             outer_manual.predict(outer_X_validation),
                         ))})
    checks = [legacy_parity, outer_parity]
    parity = {
        "rows": sum(check["rows"] for check in checks),
        "maximum_absolute_probability_difference": max(
            check["maximum_absolute_probability_difference"] for check in checks
        ),
        "tolerance": legacy_parity["tolerance"],
        "passed": all(check["passed"] for check in checks),
        "passed_predictions": all(check["passed_predictions"] for check in checks),
        "sklearn_algorithm": "brute", "checks": checks,
    }
    write_json(output_dir / "manual_v2_parity.json", parity)
    return frame, parity


def run(data=None, output_dir=OUTPUT_DIR, maximum_stage=5, resume=False, force_stage=None):
    output_dir = Path(output_dir)
    source = find_dataset(data)
    dataset = load_dataset(source)
    train, test = split_dataset(dataset)
    X, y = dataset.X.iloc[train].reset_index(drop=True), dataset.y[train]
    if output_dir.exists() and any(output_dir.iterdir()) and not resume and force_stage is None:
        raise ValueError("V2 output exists; use --resume or --force-stage N")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "reference_manifest.json"
    if not manifest_path.exists():
        create_reference_manifest(dataset, source, train, test, output_dir)
    state_path = output_dir / "search_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "completed_stages": [], "current_configuration": V2Configuration().to_dict(),
        "current_threshold": None, "decisions": [],
    }
    inner, outer = _load_frames(output_dir)
    if force_stage is not None:
        if force_stage < 1 or force_stage > 5:
            raise ValueError("--force-stage must be 1 through 5")
        state["completed_stages"] = [s for s in state["completed_stages"] if s < force_stage]
        state["decisions"] = [d for d in state["decisions"] if d["stage"] < force_stage]
        inner = inner[inner.stage < force_stage] if not inner.empty else inner
        outer = outer[outer.stage < force_stage] if not outer.empty else outer
        if state["completed_stages"]:
            state["current_configuration"] = state["decisions"][-1]["retained_configuration"]
            state["current_threshold"] = state["decisions"][-1]["next_threshold"]
        else:
            state["current_configuration"] = V2Configuration().to_dict()
            state["current_threshold"] = None
        for frozen in (output_dir / "final_v2_configuration.json",
                       output_dir / "legacy_test_comparison.csv", output_dir / "manual_v2_parity.json"):
            if frozen.exists():
                frozen.unlink()
    current = V2Configuration.from_dict(state["current_configuration"])
    for stage in range(1, maximum_stage + 1):
        if stage in state["completed_stages"]:
            continue
        candidates = stage_candidates(stage, current)
        current, threshold, stage_inner, stage_outer, decision = run_stage(
            X, y, stage, current, candidates,
        )
        decision["stage_name"] = STAGE_NAMES[stage]
        decision["retained_configuration"] = current.to_dict()
        state["completed_stages"].append(stage)
        state["decisions"].append(decision)
        state["current_configuration"] = current.to_dict()
        state["current_threshold"] = threshold
        inner = pd.concat([inner, stage_inner], ignore_index=True)
        outer = pd.concat([outer, stage_outer], ignore_index=True)
        _persist(output_dir, inner, outer, state["decisions"], state)
    if maximum_stage < 5 or state["completed_stages"] != [1, 2, 3, 4, 5]:
        return state
    final_path = output_dir / "final_v2_configuration.json"
    final = read_json(final_path) if final_path.exists() else freeze_final_configuration(
        current, state["current_threshold"], state["decisions"], outer, output_dir,
    )
    if protected_hashes() != read_json(manifest_path)["protected_hashes"]:
        raise ValueError("A protected reference changed during V2")
    legacy_path = output_dir / "legacy_test_comparison.csv"
    if not legacy_path.exists():
        evaluate_legacy(dataset, train, test, final, output_dir)
    from .reporting import create_reports
    create_reports(output_dir)
    return state
