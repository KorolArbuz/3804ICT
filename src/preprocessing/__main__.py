"""Prepare both frozen representations without fitting a classifier."""
import argparse
from pathlib import Path
from src.common.config import PROCESSED_DATA_DIR
from src.common.experiment import load_configs
from .prepared import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PROCESSED_DATA_DIR / "final/data"
    )
    args = parser.parse_args(argv)
    _, manifest = prepare(args.data, args.output_dir, load_configs())
    print({name: value["test_shape"] for name, value in manifest["groups"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
