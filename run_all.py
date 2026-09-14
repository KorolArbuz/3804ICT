import argparse
import os
from pathlib import Path
import sys

from src.common.config import DEFAULT_BATCH_SIZE, PROCESSED_DATA_DIR, RESULTS_DIR


def run_stage(name, function, *args, **kwargs):
    print(f"\n{name}", flush=True)
    return function(*args, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the complete Custom, scikit-learn, and Weka KNN experiment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the UCI credit-default XLS, XLSX, or CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Query batch size used by the custom NumPy classifier",
    )
    parser.add_argument(
        "--skip-weka",
        action="store_true",
        help="Run only the two Python implementations",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the Java/Weka JAR with Maven",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for variable in thread_variables:
        os.environ[variable] = "1"

    # Import numerical libraries after limiting their worker threads.
    from src.custom_knn.runner import run as run_custom
    from src.data.dataset import load_dataset
    from src.evaluation.comparison import compare
    from src.preprocessing.export import export_arff, preprocess
    from src.reporting.plots import plot_results
    from src.sklearn_knn.runner import run as run_sklearn
    from src.tuning.search import tune
    from src.weka_bridge.runner import run as run_weka

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = run_stage("1. Load and validate dataset", load_dataset, args.data)
    print(f"Validated {len(dataset.y)} rows and {dataset.X.shape[1]} predictors", flush=True)

    selected_path = RESULTS_DIR / "selected_parameters.json"
    selected = run_stage(
        "2. Select k by training-only cross-validation",
        tune,
        args.data,
        RESULTS_DIR / "cv_results.csv",
        selected_path,
    )
    run_stage(
        "3. Fit preprocessing on the training set",
        preprocess,
        args.data,
        PROCESSED_DATA_DIR,
        selected_path,
    )

    train_path = PROCESSED_DATA_DIR / "train.npz"
    test_path = PROCESSED_DATA_DIR / "test.npz"
    run_stage(
        "4. Export the shared Weka ARFF files",
        export_arff,
        train_path,
        test_path,
        PROCESSED_DATA_DIR,
    )

    selected_k = selected["selected_k"]
    run_stage(
        "5. Run the custom NumPy KNN",
        run_custom,
        train_path,
        test_path,
        RESULTS_DIR / "predictions_custom.csv",
        selected_k,
        args.batch_size,
        selected_path,
    )
    run_stage(
        "6. Run scikit-learn KNN",
        run_sklearn,
        train_path,
        test_path,
        RESULTS_DIR / "predictions_sklearn.csv",
        selected_k,
        selected_path,
    )
    implementations = ["custom", "sklearn"]
    exit_code = 0
    if not args.skip_weka:
        try:
            run_stage(
                "7. Build and run Java/Weka IBk",
                run_weka,
                PROCESSED_DATA_DIR / "train.arff",
                PROCESSED_DATA_DIR / "test.arff",
                selected_k,
                RESULTS_DIR / "predictions_weka.csv",
                selected_path,
                args.force_rebuild,
            )
            implementations.append("weka")
        except Exception as error:
            print(f"Weka failed: {error}", file=sys.stderr, flush=True)
            exit_code = 1

    run_stage(
        "8. Compare predictions and calculate metrics",
        compare,
        RESULTS_DIR,
        train_path,
        test_path,
        implementations,
        selected_path,
    )
    run_stage("9. Generate charts", plot_results, RESULTS_DIR)

    if len(implementations) == 3:
        status = "complete"
    else:
        status = "Python implementations complete"
    print(f"\nExperiment {status}. Results: {RESULTS_DIR}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
