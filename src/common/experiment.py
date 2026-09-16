"""Portable provenance, frozen configurations and shared experiment contracts."""
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .utils import read_json, write_json

IMPLEMENTATIONS = (
    "baseline_python_v5_1",
    "final_python",
    "final_sklearn",
    "final_cpp",
    "final_weka",
)
SCORE_TOLERANCE = 1e-12  # Declared before comparison; toolkit differences are reported.
BASELINE_SHA256 = "7d1a675e7a0b15e4e43ff521cd2137e0481a7f58c5a70557c9c45dd2b5929652"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def array_hash(value):
    """SHA-256(header JSON + LF + contiguous little-endian bytes)."""
    array = np.asarray(value)
    if array.dtype.kind not in "bif":
        return semantic_hash(
            {"shape": list(array.shape), "strings": array.astype(str).tolist()}
        )
    dtype = array.dtype.newbyteorder("<")
    array = np.ascontiguousarray(array, dtype=dtype)
    header = json.dumps(
        {"dtype": dtype.str, "shape": list(array.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(header.encode() + b"\n" + array.tobytes()).hexdigest()


def load_configs():
    result = {}
    for group in ("baseline", "final"):
        config = read_json(PROJECT_ROOT / "configs" / f"{group}.json")
        if config["schema_version"] != 1 or config["model_group"] != group:
            raise ValueError("Unknown frozen configuration schema/group")
        if config["config_hash"] != semantic_hash(config["model"]):
            raise ValueError("Frozen configuration semantic hash differs")
        expected = (
            (19, "uniform", None)
            if group == "baseline"
            else (101, "distance", 0.3315411365543412)
        )
        if (
            config["model"]["k"],
            config["model"]["weights"],
            config["model"]["threshold"],
        ) != expected:
            raise ValueError("This runner implements only the accepted frozen models")
        result[group] = config
    return result


def model_group(implementation):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown implementation: {implementation}")
    return "baseline" if implementation == IMPLEMENTATIONS[0] else "final"


def source_hashes():
    paths = [
        PROJECT_ROOT / "run_all.py",
        PROJECT_ROOT / "CMakeLists.txt",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "weka/pom.xml",
    ]
    for folder in ("src", "tests", "configs", "weka/src", "scripts"):
        paths.extend(
            p
            for p in (PROJECT_ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".cpp", ".hpp", ".java", ".json"}
        )
    return {
        p.relative_to(PROJECT_ROOT).as_posix(): sha256(p)
        for p in sorted(set(paths))
        if "__pycache__" not in p.parts
    }


def environment():
    from importlib.metadata import version
    from threadpoolctl import threadpool_info

    pools = [
        {k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            p: version(p)
            for p in (
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "threadpoolctl",
            )
        },
        "thread_environment": {k: os.environ.get(k) for k in THREAD_VARIABLES},
        "threadpools": pools,
        "affinity": "not changed",
        "priority": "not changed",
    }


def markdown_table(frame):
    """No optional tabulate dependency."""

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(map(cell, row)) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )
