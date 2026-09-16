"""Command-line entry point for Model Quality V2."""

import argparse
from pathlib import Path

from .search import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="UCI credit-default XLS/XLSX/CSV")
    parser.add_argument("--stage", type=int, choices=range(1, 6), default=5,
                        help="Stop after this stage (default: 5 and legacy comparison)")
    parser.add_argument("--resume", action="store_true", help="Continue after completed stages")
    parser.add_argument("--force-stage", type=int, choices=range(1, 6),
                        help="Recompute this stage and every later stage")
    args = parser.parse_args(argv)
    run(args.data, maximum_stage=args.stage, resume=args.resume, force_stage=args.force_stage)


if __name__ == "__main__":
    main()
