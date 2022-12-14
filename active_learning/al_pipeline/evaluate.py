"""Evaluate a series of learned models.
"""

import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator

from active_learning import stat_helper
from active_learning.al_pipeline.learn import update
from active_learning.al_pipeline.helpers import (
    Params,
    Pool,
    IndividualOutputDataContainer,
    OutputHelper,
)


def get_report(
    learner: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    training_data: int,
    i: int,
    path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate the learner's performance on a particular set of data.

    Parameters
    ----------
    learner : BaseEstimator
        Learner to evaluate
    X : np.ndarray
        Features to evaluate on
    y : np.ndarray
        Labels for the features
    training_data : int
        Amount of training data model has seen
    iteration : int
        The iteration of active learning
    path : Path
        File to save the report to, defaults to None. If None, will not save the report.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        The learner's predictions on the set and the report of the learner's performance
    """

    if X is None or y is None or X.shape[0] <= 0 or y.shape[0] <= 0:
        return None, None

    preds = learner.predict(X)

    report = metrics.classification_report(y, preds, zero_division=1, output_dict=True)
    report = {k.replace(" ", "_"): v for k, v in report.items()}
    report["iteration"] = i
    report["training_data"] = training_data
    report["accuracy"] = metrics.accuracy_score(y, preds)
    report["hamming_loss"] = metrics.hamming_loss(y, preds)

    if path is not None:
        with open(path / f"{str(i)}.json", "w", encoding="utf8") as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(",", ": "))

    return preds, report


def evaluate(container: IndividualOutputDataContainer) -> None:
    """Perform inference upon a container, where trained models can be located.

    Parameters
    ----------
    container : IndividualOutputDataContainer
        A container to contain the various paths and files produced by the experiment.
    """
    print("-" * 80, flush=True)
    start = time.time()
    print("0:00:00 -- Starting Evaluation", flush=True)

    unlabeled_pool = Pool(
        X_path=container.X_unlabeled_pool_file, y_path=container.y_unlabeled_pool_file
    ).load()
    test_set = Pool(X_path=container.X_test_set_file, y_path=container.y_test_set_file).load()

    unlabeled_pool_init_size = unlabeled_pool.y.shape[0]
    training_data = 0
    n_iterations = sum(1 for _ in container.model_path.iterdir())
    for i in range(n_iterations):
        model = joblib.load(container.model_path / f"{i}.joblib")

        idx = np.load(container.batch_path / f"{i}.npy")
        unlabeled_pool.X = stat_helper.remove_ids_from_array(unlabeled_pool.X, idx)
        unlabeled_pool.y = stat_helper.remove_ids_from_array(unlabeled_pool.y, idx)
        training_data += len(idx)

        get_report(
            model,
            unlabeled_pool.X,
            unlabeled_pool.y,
            training_data,
            i,
            container.raw_train_set_path,
        )
        get_report(
            model,
            test_set.X,
            test_set.y,
            training_data,
            i,
            container.raw_test_set_path,
        )

        update(start, unlabeled_pool_init_size, unlabeled_pool.y.shape[0], idx.shape[0], i)

    diff = datetime.timedelta(seconds=(round(time.time() - start)))
    print(f"{diff} -- Ending Evaluation", flush=True)
    print("-" * 80, flush=True)


def main(params: Params) -> None:
    """Evaluate saved model from an AL experiment for a set of experiment parmameters.

    Parameters
    ----------
    params : Params
        Experiment parameters.
    """
    oh = OutputHelper(params)
    evaluate(oh.container)
