import argparse
from pathlib import Path

from .evaluate import evaluate
from .search import search


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the separate training-only KNN model-quality experiment."
    )
    parser.add_argument("--data", type=Path, help="UCI XLS/XLSX/CSV path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--skip-weka", action="store_true")
    args = parser.parse_args(argv)

    keyword = {} if args.output_dir is None else {"output_dir": args.output_dir}
    search(args.data, **keyword)
    if not args.search_only:
        evaluate(args.data, skip_weka=args.skip_weka, **keyword)


if __name__ == "__main__":
    main()
