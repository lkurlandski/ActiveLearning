"""Run the active learning process.

TODO
----
- Refector the main method to reduce its length and complexity.
- Implement a label-encoding strategy for the target array.

FIXME
-----
-
"""

import datetime
import json
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import random
import sys  # pylint: disable=unused-import
import time
from typing import Any, Dict, Tuple, Union
import warnings

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

from active_learning import estimators
from active_learning import output_helper
from active_learning import stat_helper
from active_learning import stopping_methods
from active_learning import dataset_fetchers
from active_learning import feature_extractors


query_strategies = {
    "entropy_sampling": entropy_sampling,
    "margin_sampling": margin_sampling,
    "uncertainty_batch_sampling": uncertainty_batch_sampling,
    "uncertainty_sampling": uncertainty_sampling,
}


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


def report_to_json(
    report: Dict[str, Union[float, Dict[str, float]]], report_path: Path, i: int
) -> None:
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


def get_first_batch(y: Union[np.ndarray, sparse.csr_matrix], protocol: str, k: int) -> np.ndarray:
    """Get indices for the first batch of AL.

    Parameters
    ----------
    y : Union[np.ndarray, sparse.csc_matrix]
        The target vector for the examples which are available for selection.
    protocol : str
        String describing how to select examples. One of {"random", "k_per_class"}.
    k : int
        Determines how many examples are selected. If protocol is "random", k is the absolute number
            of elements selected. If protocol is "k_per_class", k is the number of instances
            belonging to each class that are selected.

    Returns
    -------
    np.ndarray
        Indices for the first batch.

    Raises
    ------
    TypeError
        If y is not of the correct data type.
    ValueError
        If protocol is not recognized.
    """

    if protocol == "random":
        idx = np.array(random.sample(list(range(y.shape[0])), k))

        return idx

    elif protocol == "k_per_class":
        idx = {l: [] for l in unique_labels(y)}
        for i, label in enumerate(y):

            if isinstance(y, np.ndarray) and y.ndim == 1:
                positive_classes = [int(label)]
            elif isinstance(y, np.ndarray) and y.ndim == 2:
                positive_classes = np.where(label == 1)[0].tolist()
            elif sparse.isspmatrix_csr(label):
                positive_classes = label.indices.tolist()
            else:
                raise TypeError(
                    f"Encountered an unexpected type, {type(label)},"
                    f" when iterating through y, {type(y)}."
                )

            for c in positive_classes:
                if len(idx[c]) < k:
                    idx[c].append(i)

        idx = list(idx.values())
        idx = np.array(idx).flatten()

        return idx

    raise ValueError(f"protocol: {protocol} not recognized.")


def evaluate_and_record(
    learner: ActiveLearner, X: np.ndarray, y: np.ndarray, raw_path: Path, iteration: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evlaluate and record the learner's performance on a particular set of data.

    Parameters
    ----------
    learner : ActiveLearner
        Learner to evaluate
    X : np.ndarray
        Features to evaluate on
    y : np.ndarray
        Labels for the features
    raw_path : Path
        Where to save the results (corresponds to report_to_json())
    iteration : int
        The iteration of AL

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        The learner's predictions on the set and the report of the learner's performance
    """

    # Evaluate the learner on the unlabeled pool
    preds = np.array([])
    report = {}
    if y.shape[0] > 0:
        preds = learner.predict(X)
        report = metrics.classification_report(y, preds, zero_division=1, output_dict=True)
        report = {k.replace(" ", "_"): v for k, v in report.items()}
        if "accuracy" not in report:
            report["accuracy"] = metrics.accuracy_score(y, preds)
        report["hamming_loss"] = metrics.hamming_loss(y, preds)
        report_to_json(report, raw_path, iteration)

    return preds, report


def main(experiment_parameters: Dict[str, Union[str, int]]) -> None:
    """Run the active learning algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    start = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(start - start))))} -- Preparing AL")

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
    # And some stuff from random
    random.seed(random_state)

    # Get the dataset
    X_unlabeled_pool, X_test, y_unlabeled_pool, y_test, _ = dataset_fetchers.get_dataset(
        experiment_parameters["dataset"], stream=True, random_state=random_state
    )
    # Perform the feature extraction
    X_unlabeled_pool, X_test = feature_extractors.get_features(
        X_unlabeled_pool, X_test, experiment_parameters["feature_representation"]
    )
    unlabeled_pool_initial_size = y_unlabeled_pool.shape[0]

    # Get the batch size (handle proportions and absolute values)
    tmp = float(experiment_parameters["batch_size"])
    batch_size = int(tmp) if tmp.is_integer() else int(tmp * unlabeled_pool_initial_size)
    batch_size = max(1, batch_size)

    # Get the stop set size (handle proportions and absolute values)
    tmp = float(experiment_parameters["stop_set_size"])
    stop_set_size = int(tmp) if tmp.is_integer() else int(tmp * unlabeled_pool_initial_size)
    stop_set_size = max(1, stop_set_size)

    # Select a stop set for stabilizing predictions
    stop_set_size = min(stop_set_size, unlabeled_pool_initial_size)
    stop_set_idx = rng.choice(unlabeled_pool_initial_size, size=stop_set_size, replace=False)
    X_stop_set, y_stop_set = X_unlabeled_pool[stop_set_idx], y_unlabeled_pool[stop_set_idx]

    # Get the modAL compatible estimator and query strategy to use
    estimator = estimators.get_estimator(
        base_learner_code=experiment_parameters["base_learner"],
        y=y_unlabeled_pool,
        multiclass_code=experiment_parameters["multiclass"],
    )
    query_strategy = query_strategies[experiment_parameters["query_strategy"]]

    # Setup output directory structure
    oh = output_helper.OutputHelper(experiment_parameters)
    oh.setup()

    # Track the number of training data in the labeled pool
    training_data = []

    # Guard had some complex boolean algebra, but is correct
    i = 0
    print(f"{str(datetime.timedelta(seconds=(round(time.time() - start))))} -- Starting AL Loop")
    while y_unlabeled_pool.shape[0] > 0 and not (
        stopping_manager.stopping_condition_met(stopping_condition) and early_stopping_enabled
    ):

        # Setup the learner and stabilizing predictions in the 0th iteration
        if i == 0:
            idx = get_first_batch(y_unlabeled_pool, protocol="random", k=batch_size)
            learner = ActiveLearner(estimator=estimator, query_strategy=query_strategy)
            learner.teach(X_unlabeled_pool[idx], y_unlabeled_pool[idx])

        # Retrain the learner during every other iteration
        else:
            if batch_size > y_unlabeled_pool.shape[0]:
                batch_size = y_unlabeled_pool.shape[0]
            idx, query_sample = learner.query(X_pool=X_unlabeled_pool, n_instances=batch_size)
            query_labels = y_unlabeled_pool[idx]
            learner.teach(query_sample, query_labels)

        # Remove the queried elements from the unlabeled pool
        X_unlabeled_pool = stat_helper.remove_ids_from_array(X_unlabeled_pool, idx)
        y_unlabeled_pool = stat_helper.remove_ids_from_array(y_unlabeled_pool, idx)

        _, _ = evaluate_and_record(
            learner, X_unlabeled_pool, y_unlabeled_pool, oh.container.raw_train_set_path, i
        )
        preds_stop_set, _ = evaluate_and_record(
            learner, X_stop_set, y_stop_set, oh.container.raw_stop_set_path, i
        )
        _, report_test_set = evaluate_and_record(
            learner, X_test, y_test, oh.container.raw_test_set_path, i
        )

        # Evaluate the stopping methods
        stopping_manager.check_stopped(preds=preds_stop_set)
        stopping_manager.update_results(
            **{
                "annotations": learner.y_training.shape[0],
                "iteration": i,
                "accuracy": report_test_set["accuracy"],
            }
        )

        # Print a status update and increment the iteration counter
        print_update(
            start, unlabeled_pool_initial_size, y_unlabeled_pool, i, report_test_set["accuracy"]
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

    from active_learning import local

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(local.experiment_parameters)
