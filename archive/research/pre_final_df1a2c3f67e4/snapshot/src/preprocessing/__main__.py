import argparse
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR

from .export import export_arff, export_cpp_csv, preprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess the final train and test partitions.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--selected-parameters", type=Path)
    parser.add_argument(
        "--cpp-output-dir",
        type=Path,
        help="Also export canonical headerless CSV inputs for the C++ KNN",
    )
    args = parser.parse_args(argv)
    preprocess(args.data, args.output_dir, args.selected_parameters)
    export_arff(args.output_dir / "train.npz", args.output_dir / "test.npz", args.output_dir)
    if args.cpp_output_dir is not None:
        export_cpp_csv(
            args.output_dir / "train.npz",
            args.output_dir / "test.npz",
            args.cpp_output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
