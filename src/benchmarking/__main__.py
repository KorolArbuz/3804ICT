"""Run the same coordinated benchmark as run_all.py --benchmark."""
import sys
from src.experiment import main

if __name__ == "__main__":
    raise SystemExit(main(["--benchmark", *sys.argv[1:]]))
