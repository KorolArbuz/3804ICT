import argparse
from pathlib import Path

from src.common.config import RESULTS_DIR

from .search import tune


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select k using training-only cross-validation.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "cv_results.csv")
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=RESULTS_DIR / "selected_parameters.json",
    )
    args = parser.parse_args(argv)
    tune(args.data, args.output, args.selected_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
