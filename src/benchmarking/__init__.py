"""Reproducible runtime benchmarking and reporting.

The environment is constrained before NumPy or scikit-learn are imported by
benchmark submodules.  ``threadpoolctl`` is also used around measured work.
"""

import os


SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

os.environ.update(SINGLE_THREAD_ENVIRONMENT)
