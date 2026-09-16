import argparse
from pathlib import Path

from src.common.config import RESULTS_DIR
from src.common.utils import write_json

from .dataset import load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect and validate the raw dataset.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "dataset_summary.json")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.data)
    write_json(args.summary, dataset.summary)
    print(f"Validated {len(dataset.y)} rows, {dataset.X.shape[1]} predictors -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
