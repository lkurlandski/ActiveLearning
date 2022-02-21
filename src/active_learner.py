"""Run the active learning process.

TODO: may contain some sort of bug that causes signficant dips in accuracy, unexpectedly.
"""

import json
import math
from pathlib import Path
from pprint import pprint
from random import choices
from typing import Any, Dict, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import cohen_kappa_score, classification_report

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner

import estimators
import input_helper
import output_helper
import stat_helper

def print_pool_update(
        unlabeled_pool_initial_size:int, 
        y_unlabeled_pool:np.ndarray, 
        iteration:int
    ) -> None:
    """Print an update of AL progress.

    Parameters
    ----------
    unlabeled_pool_initial_size : int
        Size of the unlabeled pool before the iterative process.
    y_unlabeled_pool : np.ndarray
        The current unlabeled pool.
    iteration : int
        The current iteration of AL.
    """

    prop = (unlabeled_pool_initial_size - y_unlabeled_pool.shape[0]) / unlabeled_pool_initial_size

    print(
        f"Iteration {iteration} : {round(100 * prop, 2)}%:"
        f"\t|U_0|:{unlabeled_pool_initial_size}"
        f"\t|U|:{y_unlabeled_pool.shape[0]}"
        f"\t|L|:{unlabeled_pool_initial_size - y_unlabeled_pool.shape[0]}",
        flush=True
    )

def get_index_for_each_class(y:np.ndarray, labels:np.ndarray) -> np.ndarray:
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

    # TODO: come up with a better label-encoding strategy
    # If the data is label-encoded (as it should be) but the labels are raw strings
        # (as they perhaps should be) this ensures one of every encoded label is captured
    if y[0] not in labels:
        labels = [i for i in range(len(labels))]

    idx = {l : None for l in labels}
    for i, l in enumerate(y):
        if idx[l] is None:
            idx[l] = i
    idx = [i for i in idx.values()]

    return idx

def main(experiment_parameters:Dict[str, Union[str, int]]) -> None:
    """Run the active learning algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    print("active_learner called with experiment_parameters:")
    pprint(experiment_parameters)

    # Extract hyperparameters from the experiment parameters
    stop_set_size = int(experiment_parameters["stop_set_size"])
    batch_size = int(experiment_parameters["batch_size"])
    estimator = estimators.get_estimator_from_code(experiment_parameters["estimator"])
    random_state = int(experiment_parameters["random_state"])

    # Get the dataset and select a stop set from it
    X_unlabeled_pool, X_test, y_unlabeled_pool, y_test, labels = \
        input_helper.get_dataset(experiment_parameters["dataset"], random_state)
    unlabeled_pool_initial_size = y_unlabeled_pool.shape[0]
    stop_set_idx = choices([i for i in range(len(y_unlabeled_pool))], k=stop_set_size)
    X_stop_set, y_stop_set = X_unlabeled_pool[stop_set_idx], y_unlabeled_pool[stop_set_idx]

    # Setup output directory structure
    oh = output_helper.OutputHelper(experiment_parameters)
    oh.setup_output_path(remove_existing=True)
    print(f"output_path:{oh.output_path.as_posix()}\n")

    # Get ids of one instance of each class
    idx = get_index_for_each_class(y_unlabeled_pool, labels)

    # Write the size of the unlabeled pool at each iteration to a file
    n_iterations = math.ceil(y_unlabeled_pool.shape[0] / batch_size) + 1
    pd.DataFrame({'training_data' : 
        [len(idx) + i * batch_size for i in range(n_iterations - 1)] + [y_unlabeled_pool.shape[0]]
    }).to_csv(oh.ind_rstates_paths['num_training_data_file'])

    # These assets will be initialized in the 0th iteration of AL
    learner, previous_stop_set_predictions, kappas = None, None, None

    i = -1
    while y_unlabeled_pool.shape[0] > 0:

        i += 1

        # Setup the learner and stabilizing predictions in the 0th iteration.
        if i == 0:
            learner = ActiveLearner(estimator=estimator, query_strategy=uncertainty_batch_sampling)
            learner.teach(X_unlabeled_pool[idx], y_unlabeled_pool[idx])
            previous_stop_set_predictions = learner.predict(X_unlabeled_pool[stop_set_idx])
            kappas = [np.NaN]

        # Retrain the learner
        else:
            # FIXME: remove this error handling once bug fixed
            try:
                idx, query_sample = learner.query(X_pool=X_unlabeled_pool, n_instances=batch_size)
                query_labels = y_unlabeled_pool[idx]
                learner.teach(query_sample, query_labels)
            except ValueError as e:
                print(f"Caught and ignoring the following exception:\n\t{e}")

        # Remove the queried elements from the unlabeled pool
        X_unlabeled_pool = stat_helper.remove_ids_from_array(X_unlabeled_pool, idx)
        y_unlabeled_pool = stat_helper.remove_ids_from_array(y_unlabeled_pool, idx)

        # Print the progress of the learning procedure
        print_pool_update(unlabeled_pool_initial_size, y_unlabeled_pool, i)

        # Evaluate the learner on the test set
        predictions = learner.predict(X_test)
        report = classification_report(
            y_test, predictions, zero_division=1, output_dict=True)
        with open(oh.ind_rstates_paths["report_test_path"] / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

        # Evaluate the learner on the unlabeled pool
        if y_unlabeled_pool.shape[0] > 0:
            predictions = learner.predict(X_unlabeled_pool)
            report = classification_report(
                y_unlabeled_pool, predictions, zero_division=1, output_dict=True)
            with open(oh.ind_rstates_paths["report_train_path"] / f"{str(i)}.json", 'w') as f:
                json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

        # Evaluate the learner on the stop set
        stop_set_predictions = learner.predict(X_stop_set)
        report = classification_report(
            y_stop_set, stop_set_predictions, zero_division=1, output_dict=True)
        with open(oh.ind_rstates_paths["report_stop_set_path"] / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

        # Update the list of kappas for stabilizing predictions
        kappa = cohen_kappa_score(stop_set_predictions, previous_stop_set_predictions)
        kappas.append(kappa)
        previous_stop_set_predictions = stop_set_predictions

    np.savetxt(oh.ind_rstates_paths["kappa_file"], np.array(kappas), fmt='%f', delimiter=',')

if __name__ == "__main__":

    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 7,
        "estimator": "svm",
        "dataset": "Iris",
        "random_state": 0,
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(experiment_parameters)
