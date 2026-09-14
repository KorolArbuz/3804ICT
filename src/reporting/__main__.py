import argparse
from pathlib import Path

from src.common.config import RESULTS_DIR

from .plots import plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate charts from the result tables.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    plot_results(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
