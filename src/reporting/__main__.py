import argparse
from pathlib import Path

from .plots import plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate final figures and tables from one saved run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = plot_results(args.run_dir)
    print(f"Generated {len(report['figures']) // 2} PNG/SVG figures for {report['run_id']}")
    for omission in report["omissions"]:
        print(omission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
