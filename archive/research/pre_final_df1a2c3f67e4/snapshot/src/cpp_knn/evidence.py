"""Shared evidence helpers for the C++ candidate verification and benchmark."""

import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_hash(labels, positive_vote_counts):
    """Match the executable's FNV-1a hash over index,label,positive_votes rows."""
    if len(labels) != len(positive_vote_counts):
        raise ValueError("Hash inputs have different lengths")
    value = 14695981039346656037
    for index, (label, votes) in enumerate(zip(labels, positive_vote_counts)):
        row = f"{index},{int(label)},{int(votes)}\n".encode("ascii")
        for byte in row:
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"fnv1a64:{value:016x}"


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize empty timings")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def timing_summary(values):
    values = [float(value) for value in values]
    return {
        "raw_seconds": values,
        "median_seconds": percentile(values, 0.5),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "iqr_seconds": percentile(values, 0.75) - percentile(values, 0.25),
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")

