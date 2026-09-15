import argparse
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR
from src.preprocessing.export import export_cpp_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export canonical processed arrays for the pure C++ KNN."
    )
    parser.add_argument(
        "--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz"
    )
    parser.add_argument(
        "--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROCESSED_DATA_DIR / "cpp"
    )
    args = parser.parse_args(argv)
    export_cpp_csv(args.train, args.test, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
