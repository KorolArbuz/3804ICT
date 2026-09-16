"""Run the five frozen implementations on one fixed split."""
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
from src.experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
