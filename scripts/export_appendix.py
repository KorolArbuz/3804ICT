"""Generate a source appendix with language, path and raw-byte hashes."""

import argparse
from pathlib import Path

try:
    from .export_submission import ROOT, appendix_bytes, source_files
except ImportError:
    from export_submission import ROOT, appendix_bytes, source_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/appendix_source_code.md"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(appendix_bytes(source_files(ROOT)))
    print(f"Source appendix: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
