import argparse
from pathlib import Path

from src.common.config import PROCESSED_DATA_DIR, RESULTS_DIR

from .comparison import IMPLEMENTATIONS, compare


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare model predictions and calculate metrics.")
    parser.add_argument("--results-dir", "--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument("--selected-parameters", type=Path)
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS)
    for name in IMPLEMENTATIONS:
        parser.add_argument(f"--{name}", type=Path, help=f"Explicit {name} prediction file")
    args = parser.parse_args(argv)

    paths = {}
    for name in IMPLEMENTATIONS:
        prediction_path = getattr(args, name)
        if prediction_path is not None:
            paths[name] = prediction_path

    if args.implementations:
        implementations = args.implementations
    elif paths:
        implementations = list(paths)
    else:
        implementations = None

    compare(
        args.results_dir,
        args.train,
        args.test,
        implementations,
        args.selected_parameters,
        paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
