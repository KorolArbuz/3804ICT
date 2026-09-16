import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import PROCESSED_DATA_DIR, RESULTS_DIR
from .utils import load_pair, read_json, write_json


def runner_parser(description, implementation):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", type=Path, default=PROCESSED_DATA_DIR / "train.npz")
    parser.add_argument("--test", type=Path, default=PROCESSED_DATA_DIR / "test.npz")
    parser.add_argument(
        "--k",
        type=int,
        help="Explicit k; must agree with selection artifact when present",
    )
    parser.add_argument(
        "--selected-parameters",
        type=Path,
        default=RESULTS_DIR / "selected_parameters.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / f"predictions_{implementation}.csv",
    )
    return parser


def run_python(implementation, train, test, output, requested_k, selected_parameters, factory):
    training, testing = load_pair(train, test)
    selection_path = None
    if selected_parameters is not None:
        selection_path = Path(selected_parameters)

    if selection_path is not None and selection_path.exists():
        selection = read_json(selection_path)
        selected_k = int(selection["selected_k"])
        if requested_k is not None and requested_k != selected_k:
            raise ValueError("Explicit k disagrees with training-only selected k")
    elif requested_k is not None and requested_k >= 1:
        selected_k = requested_k
    else:
        raise ValueError(
            "Supply --k or an existing --selected-parameters; no default k is invented"
        )

    classifier = factory(selected_k)
    with threadpool_limits(limits=1):
        start = perf_counter()
        classifier.fit(training["X"], training["y"])
        fit_seconds = perf_counter() - start

        start = perf_counter()
        probabilities = classifier.predict_proba(testing["X"])
        class_indices = np.argmax(probabilities, axis=1)
        predictions = classifier.classes_[class_indices]
        prediction_seconds = perf_counter() - start

    class_one = int(np.flatnonzero(classifier.classes_ == 1)[0])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    prediction_table = pd.DataFrame({
        "test_index": np.arange(len(predictions)),
        "row_id": testing["row_ids"],
        "original_id": testing["original_ids"],
        "y_true": testing["y"],
        "y_pred": predictions,
        "probability_class_1": probabilities[:, class_one],
        "selected_k": selected_k,
    })
    prediction_table.to_csv(output, index=False, float_format="%.17g")

    milliseconds_per_sample = 1000 * prediction_seconds / len(predictions)
    timing = {
        "implementation": implementation,
        "selected_k": selected_k,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "milliseconds_per_sample": milliseconds_per_sample,
        "timing_scope": (
            "one fit; one predict_proba query plus argmax; "
            "I/O excluded; threadpool limit 1"
        ),
    }
    write_json(output.with_suffix(".runtime.json"), timing)
    print(
        f"{implementation}: k={selected_k}, {len(predictions)} predictions, "
        f"{prediction_seconds:.3f}s -> {output}",
        flush=True,
    )
    return timing
