import argparse
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR

from .export import export_arff, preprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess the final train and test partitions.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--selected-parameters", type=Path)
    args = parser.parse_args(argv)
    preprocess(args.data, args.output_dir, args.selected_parameters)
    export_arff(args.output_dir / "train.npz", args.output_dir / "test.npz", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
