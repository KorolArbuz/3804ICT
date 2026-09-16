"""One fresh Python process: prepared-data load, fit, prediction and CSV output."""
import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.common.experiment import load_configs, model_group
from src.final_knn.runner import PythonModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(1):
        # CSV for each implementation, including Python, in this separate scope.
        X = np.loadtxt(args.data_dir / "train_features.csv", delimiter=",", ndmin=2)
        y = np.loadtxt(args.data_dir / "train_labels.csv", dtype=np.int64, ndmin=1)
        queries = np.loadtxt(
            args.data_dir / "test_features.csv", delimiter=",", ndmin=2
        )
        model = PythonModel(
            args.implementation, X, y, load_configs()[model_group(args.implementation)]
        )
        scores, labels = model.predict(queries)
        np.savetxt(
            args.output,
            np.column_stack((np.arange(len(scores)), scores, labels)),
            delimiter=",",
            fmt=["%d", "%.17g", "%d"],
            header="test_position,score_class_1,y_pred",
            comments="",
        )


if __name__ == "__main__":
    main()
