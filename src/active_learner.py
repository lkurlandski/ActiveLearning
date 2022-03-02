"""Run the active learning process.
"""

import datetime
import json
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import time
from typing import Dict, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner

import estimators
import input_helper
import output
import stat_helper
import stopping_methods
import vectorize


def print_update(
    start_time: float,
    unlabeled_pool_initial_size: int,
    y_unlabeled_pool: np.ndarray,
    iteration: int,
    accuracy: float,
) -> None:
    """Print an update of AL progress.

    Parameters
    ----------
    start_time : float
        Time which the AL main method began
    unlabeled_pool_initial_size : int
        Size of the unlabeled pool before the iterative process
    y_unlabeled_pool : np.ndarray
        The current unlabeled pool
    iteration : int
        The current iteration of AL
    accuracy : float
        The current accuracy of the model on a test set
    """

    prop = (unlabeled_pool_initial_size - y_unlabeled_pool.shape[0]) / unlabeled_pool_initial_size

    pool_zfill = len(str(unlabeled_pool_initial_size))

    print(
        f"itr: {str(iteration).zfill(4)}"
        f"    pgs: {str(round(100 * prop, 2)).zfill(5)}%"
        f"    time: {str(datetime.timedelta(seconds=(round(time.time() - start_time))))}"
        f"    acy: {round(accuracy, 3)}"
        f"    |U_0|: {unlabeled_pool_initial_size}"
        f"    |U|: {str(y_unlabeled_pool.shape[0]).zfill(pool_zfill)}"
        f"    |L|: {str(unlabeled_pool_initial_size - y_unlabeled_pool.shape[0]).zfill(pool_zfill)}",
        flush=True,
    )


def report_to_json(report: Dict[str, Union[float, Dict[str, float]]], report_path: Path, i: int):
    """Write a report taken from active learning to a specially named json output path.

    Parameters
    ----------
    report : Dict[str, Union[float, Dict[str, float]]]
        The report like the dict returned by sklearn.metrics.classification_report
    report_path : Path
        Directory to store the json file
    i : int
        The iteration of active learning, used the name the json file
    """

    with open(report_path / f"{str(i)}.json", "w", encoding="utf8") as f:
        json.dump(report, f, sort_keys=True, indent=4, separators=(",", ": "))


# TODO: come up with a better label-encoding strategy
def get_index_for_each_class(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return indices that contain location of one element per class.

    Parameters
    ----------
    y : np.ndarray
        Array of train/test class labels that corresponds to an array of train/test data.
    labels : np.ndarray
        Array of all available classes in the labels.

    Returns
    -------
    np.ndarray
        Indices that contain location of one element per class from y.
    """

    # If the data is label-encoded (as it should be) but the labels are raw strings
    # (as they perhaps should be) this ensures one of every encoded label is captured
    labels = list(range(len(labels))) if y[0] not in labels else labels

    idx = {l: None for l in labels}
    for i, l in enumerate(y):
        if idx[l] is None:
            idx[l] = i

    idx = list(idx.values())

    return idx


# TODO: refactor this long function to improve code quality
# TODO: instead of deleting elements from the unlabeled pool, use masked arrays for efficiency
def main(experiment_parameters: Dict[str, Union[str, int]]) -> None:
    """Run the active learning algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    start = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(start - start))))} -- Beginning Active Learning")

    # Extract hyperparameters from the experiment parameters
    stop_set_size = int(experiment_parameters["stop_set_size"])
    batch_size = int(experiment_parameters["batch_size"])
    estimator = estimators.get_estimator_from_code(experiment_parameters["estimator"])
    random_state = int(experiment_parameters["random_state"])

    # Determine a stopping condition to wait upon
    stopping_condition = stopping_methods.StabilizingPredictions(windows=3, threshold=0.99)
    early_stopping_enabled = False
    # Set up the stopping method manager
    stopping_manager = stopping_methods.Manager(
        [
            stopping_methods.StabilizingPredictions(windows=w, threshold=t)
            for w in [2, 3, 4]
            for t in [0.97, 0.98, 0.99]
        ]
        + [stopping_condition]
    )

    # To attain reproducibility with modAL, we need to use legacy numpy random seeding code
    np.random.seed(random_state)
    # Otherwise, we use the most up-to-date methods provided by numpy
    rng = np.random.default_rng(random_state)

    # Get the dataset and select a stop set from it
    X_unlabeled_pool, X_test, y_unlabeled_pool, y_test, labels = input_helper.get_dataset(
        experiment_parameters["dataset"], random_state
    )
    X_test, X_unlabeled_pool = vectorize.w2v_vectorize(X_test, X_unlabeled_pool)
    unlabeled_pool_initial_size = y_unlabeled_pool.shape[0]

    # Select a stop set for stabilizing predictions
    stop_set_size = min(stop_set_size, unlabeled_pool_initial_size)
    stop_set_idx = rng.choice(unlabeled_pool_initial_size, size=stop_set_size, replace=False)
    X_stop_set, y_stop_set = X_unlabeled_pool[stop_set_idx], y_unlabeled_pool[stop_set_idx]

    # Setup output directory structure
    oh = output.OutputHelper(experiment_parameters)
    oh.setup()

    # Get ids of one instance of each class
    idx = get_index_for_each_class(y_unlabeled_pool, labels)
    #print("this is idx", idx)
    #print(y_unlabeled_pool[idx])
    #print(X_unlabeled_pool[idx])

    # Track the number of training data in the labeled pool
    training_data = []

    # Guard had some complex boolean algebra, but is correct
    i = 0
    while y_unlabeled_pool.shape[0] > 0 and not (
        stopping_manager.stopping_condition_met(stopping_condition) and early_stopping_enabled
    ):

        # Setup the learner and stabilizing predictions in the 0th iteration.
        # Setup the learner and stabilizing predictions in the 0th iteration
        if i == 0:
            learner = ActiveLearner(estimator=estimator, query_strategy=uncertainty_batch_sampling)
            learner.teach(X_unlabeled_pool[idx], y_unlabeled_pool[idx])

        # Retrain the learner during every other iteration
        else:
            idx, query_sample = learner.query(X_pool=X_unlabeled_pool, n_instances=batch_size)
            query_labels = y_unlabeled_pool[idx]
            learner.teach(query_sample, query_labels)

        # Remove the queried elements from the unlabeled pool
        X_unlabeled_pool = stat_helper.remove_ids_from_array(X_unlabeled_pool, idx)
        y_unlabeled_pool = stat_helper.remove_ids_from_array(y_unlabeled_pool, idx)

        # Evaluate the learner on the test set
        preds_test = learner.predict(X_test)
        report_test = classification_report(y_test, preds_test, zero_division=1, output_dict=True)
        report_to_json(report_test, oh.container.raw_test_set_path, i)

        # Evaluate the learner on the unlabeled pool
        if y_unlabeled_pool.shape[0] > 0:
            preds_unlabeled_pool = learner.predict(X_unlabeled_pool)
            report_unlabeled_pool = classification_report(
                y_unlabeled_pool, preds_unlabeled_pool, zero_division=1, output_dict=True
            )
            report_to_json(report_unlabeled_pool, oh.container.raw_train_set_path, i)

        # Evaluate the learner on the stop set
        preds_stop_set = learner.predict(X_stop_set)
        report_stop_set = classification_report(
            y_stop_set, preds_stop_set, zero_division=1, output_dict=True
        )
        report_to_json(report_stop_set, oh.container.raw_stop_set_path, i)

        # Evaluate the stopping methods
        stopping_manager.check_stopped(stop_set_predictions=preds_stop_set)
        results = {
            "annotations": learner.y_training.shape[0],
            "iteration": i,
            "accuracy": report_test["accuracy"],
            "macro avg f1-score": report_test["macro avg"]["f1-score"],
            "weighted avg f1-score": report_test["weighted avg"]["f1-score"],
        }
        stopping_manager.update_results(**results)

        # Print a status update and increment the iteration counter
        print_update(
            start, unlabeled_pool_initial_size, y_unlabeled_pool, i, report_test["accuracy"]
        )

        i += 1
        training_data.append(learner.y_training.shape[0])

    # Save as both csv and json file
    stopping_manager.results_to_dataframe().to_csv(oh.container.stopping_results_csv_file)
    results = stopping_manager.results_to_dict()
    with open(oh.container.stopping_results_json_file, "w", encoding="utf8") as f:
        json.dump(results, f, sort_keys=True, indent=4, separators=(",", ": "))

    pd.DataFrame({"training_data": training_data}).to_csv(oh.container.training_data_file)

    end = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(end - start))))} -- Ending Active Learning")


if __name__ == "__main__":

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(
            experiment_parameters={
                "output_root": "/home/hpc/elphicb1/ActiveLearning/ActiveLearning/output4",
                "task": "cls",
                "stop_set_size": 1000,
                "batch_size": 10,
                "estimator": "svm-ova",
                "dataset": "Emotions",
                "random_state": 0,
            }
        )
