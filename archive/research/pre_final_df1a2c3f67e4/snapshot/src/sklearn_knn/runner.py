from sklearn.neighbors import KNeighborsClassifier

from src.common.config import RESULTS_DIR
from src.common.model_runner import run_python


def make_classifier(k):
    return KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
        n_jobs=1,
    )


def run(train, test, output, k=None, selected_parameters=RESULTS_DIR / "selected_parameters.json"):
    return run_python(
        "sklearn",
        train,
        test,
        output,
        k,
        selected_parameters,
        make_classifier,
    )
