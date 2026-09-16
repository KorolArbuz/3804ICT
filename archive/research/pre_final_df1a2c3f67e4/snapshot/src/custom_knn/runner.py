from src.common.config import DEFAULT_BATCH_SIZE, RESULTS_DIR
from src.common.model_runner import run_python

from .classifier import CustomKNNClassifier


def run(train, test, output, k=None, batch_size=DEFAULT_BATCH_SIZE,
        selected_parameters=RESULTS_DIR / "selected_parameters.json"):
    def make_classifier(selected_k):
        return CustomKNNClassifier(selected_k, batch_size)

    return run_python(
        "custom",
        train,
        test,
        output,
        k,
        selected_parameters,
        make_classifier,
    )
