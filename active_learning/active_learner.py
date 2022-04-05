"""Run the active learning process.

TODO
----
- Implement a label-encoding strategy for the target array.
- Introduce checks to ensure the base learner can accept sparse arrays and such.
- The stopping methods definitely need some refinement.

FIXME
-----
- Once the stopping methods are fixed add the stopping method back in.
"""

from dataclasses import dataclass
import datetime
import json
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import random
import sys  # pylint: disable=unused-import
import time
from typing import Any, Callable, Dict, Tuple, Union
import warnings

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import numpy as np
from scipy import sparse
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.multiclass import unique_labels, type_of_target

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


@dataclass
class Pool:
    """A pool of training examples for active learning.

    Parameters
    ----------
    X : Union[np.ndarray, sparse.spmatrix]
        Training features.
    y : Union[np.ndarray, sparse.spmatrix]
        Corresponding labels.
    path : Path
        An optional path to record evaluation data for the data, defaults to None.
    """

    X: Union[np.ndarray, sparse.spmatrix]
    y: Union[np.ndarray, sparse.spmatrix]
    path: Path = None


@dataclass
class Stopping:
    """A wrapper for the stopping metrics.

    Parameters
    ----------
    manager : stopping_methods.Manager
        A stopping method manager.
    condition : stopping_methods.StoppingMethod
        An optional condition to trigger early stopping.
    path : Path
        A path to save the stopping results to
    """

    manager: stopping_methods.Manager
    condition: stopping_methods.StoppingMethod
    path: Path


def get_update(
    start_time: float,
    unlabeled_init_size: int,
    unlabeled_size: int,
    i: int,
    accuracy: float,
) -> None:
    """Print an update of AL progress.

    Parameters
    ----------
    start_time : float
        Time which the AL main method began
    unlabeled_init_size : int
        Size of the unlabeled pool before the iterative process
    unlabeled_size : int
        The current unlabeled pool
    i : int
        The current iteration of AL
    accuracy : float
        The current accuracy of the model on a test set
    """

    prop = (unlabeled_init_size - unlabeled_size) / unlabeled_init_size

    pool_zfill = len(str(unlabeled_init_size))

    return (
        f"itr: {str(i).zfill(4)}"
        f"    pgs: {str(round(100 * prop, 2)).zfill(5)}%"
        f"    time: {str(datetime.timedelta(seconds=(round(time.time() - start_time))))}"
        f"    |U_0|: {unlabeled_init_size}"
        f"    |U|: {str(unlabeled_size).zfill(pool_zfill)}"
        f"    |L|: {str(unlabeled_init_size - unlabeled_size).zfill(pool_zfill)}",
        f"    acy: {round(accuracy, 3)}",
    )


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

    if protocol == "k_per_class":
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
    learner: ActiveLearner,
    X: np.ndarray,
    y: np.ndarray,
    training_data: int,
    i: int,
    path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate the learner's performance on a particular set of data.

    Parameters
    ----------
    learner : ActiveLearner
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


def learn(
    estimator: BaseEstimator,
    query_strategy: Callable,
    batch_size: int,
    unlabeled_pool: Pool,
    *,
    test_set: Pool = None,
    stop_set: Pool = None,
    stopping: Stopping = None,
):
    """Perform active learning.

    Parameters
    ----------
    estimator : BaseEstimator
        A modAL compatible estimator for learning and evaluating.
    query_strategy : Callable
        A modAl compatible query strategy for selecting examples.
    batch_size : int
        The number of examples to select at each iteration.
    unlabeled_pool : Pool
        An unlabeled pool of examples to query, label, and learn from. If the path
            attribute is None, the results of inference will not be saved.
    test_set : Pool, optional
        An optional test set of examples to evaluate the model upon, by default None. If the path
            attribute is None, the results of inference will not be saved.
    stop_set : Pool, optional
        An optional stop set of examples to evaluate the model upon, by default None. If the path
            attribute is None, the results of inference will not be saved.
    stopping : Stopping, optional
        A system of stopping methods for optional early stopping and stopping method evaluation.
    """

    def get_batch_size(batch_size) -> int:
        if stopping is not None and stopping.condition is not None and stopping.condition.stopped:
            batch_size = batch_size * 2
        if batch_size > unlabeled_pool.y.shape[0]:
            batch_size = unlabeled_pool.y.shape[0]
        return batch_size

    start = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(time.time() - start))))} -- Starting AL Loop")

    unlabeled_init_size = unlabeled_pool.y.shape[0]
    i = 0

    # Perform the 0th iteration of active learning
    idx = get_first_batch(unlabeled_pool.y, protocol="random", k=batch_size)
    learner = ActiveLearner(estimator=estimator, query_strategy=query_strategy)
    learner.teach(unlabeled_pool.X[idx], unlabeled_pool.y[idx])

    while unlabeled_pool.y.shape[0] > 0:

        # Retrain the learner during every iteration, except the 0th one
        if i > 0:
            batch_size = get_batch_size(batch_size)
            idx, query_sample = learner.query(unlabeled_pool.X, n_instances=batch_size)
            query_labels = unlabeled_pool.y[idx]
            learner.teach(query_sample, query_labels)

        # Remove the queried elements from the unlabeled pool
        unlabeled_pool.X = stat_helper.remove_ids_from_array(unlabeled_pool.X, idx)
        unlabeled_pool.y = stat_helper.remove_ids_from_array(unlabeled_pool.y, idx)

        # Evaluate the model on data and record its performance
        _, report_unlabeled_pool = evaluate_and_record(
            learner,
            unlabeled_pool.X,
            unlabeled_pool.y,
            learner.y_training.shape[0],
            i,
            unlabeled_pool.path,
        )
        if test_set is not None:
            _, _ = evaluate_and_record(
                learner,
                test_set.X,
                test_set.y,
                learner.y_training.shape[0],
                i,
                test_set.path,
            )
        if stop_set is not None:
            preds_stop_set, _ = evaluate_and_record(
                learner,
                stop_set.X,
                stop_set.y,
                learner.y_training.shape[0],
                i,
                stop_set.path,
            )

        # Evaluate the stopping methods
        if stopping is not None:
            stopping.manager.check_stopped(preds=preds_stop_set)
            update_results = {
                "annotations": learner.y_training.shape[0],
                "iteration": i,
            }
            stopping.manager.update_results(**update_results)

        # Perform end-of-iteration tasks
        print(
            get_update(
                start,
                unlabeled_init_size,
                unlabeled_pool.y.shape[0],
                i,
                report_unlabeled_pool["accuracy"] if report_unlabeled_pool is not None else np.NaN,
            ),
            flush=True,
        )
        i += 1

    # Save stopping results
    if stopping is not None:
        df = stopping.manager.results_to_dataframe()
        df.to_csv(stopping.path)

    # End the al experience
    end = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(end - start))))} -- Ending Active Learning")


def main(experiment_parameters: Dict[str, Union[str, int]]) -> None:
    """Run the active learning algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    random_state = int(experiment_parameters["random_state"])

    # Seed random, numpy.random, and get a numpy randomizer
    random.seed(random_state)
    np.random.seed(random_state)
    rng = np.random.default_rng(random_state)

    # Get the dataset and perform the feature extraction
    (X_unlabeled_pool, X_test, y_unlabeled_pool, y_test, _,) = dataset_fetchers.get_dataset(
        experiment_parameters["dataset"], stream=True, random_state=random_state
    )
    X_unlabeled_pool, X_test = feature_extractors.get_features(
        X_unlabeled_pool, X_test, experiment_parameters["feature_representation"]
    )
    unlabeled_init_size = y_unlabeled_pool.shape[0]

    # Get the batch size (handle proportions and absolute values)
    tmp = float(experiment_parameters["batch_size"])
    batch_size = int(tmp) if tmp.is_integer() else int(tmp * unlabeled_init_size)
    batch_size = max(1, batch_size)

    # Get the stop set size (handle proportions and absolute values)
    tmp = float(experiment_parameters["stop_set_size"])
    stop_set_size = int(tmp) if tmp.is_integer() else int(tmp * unlabeled_init_size)
    stop_set_size = max(1, stop_set_size)

    # Select a stop set for stabilizing predictions
    stop_set_size = min(stop_set_size, unlabeled_init_size)
    stop_set_idx = rng.choice(unlabeled_init_size, size=stop_set_size, replace=False)
    X_stop_set, y_stop_set = (
        X_unlabeled_pool[stop_set_idx],
        y_unlabeled_pool[stop_set_idx],
    )

    # Get the modAL compatible estimator and query strategy to use
    estimator = estimators.get_estimator(
        experiment_parameters["base_learner"],
        type_of_target(y_unlabeled_pool),
        random_state=random_state,
    )
    query_strategy = query_strategies[experiment_parameters["query_strategy"]]

    # Setup output directory structure
    oh = output_helper.OutputHelper(experiment_parameters)
    oh.setup()

    # Set up the stopping method manager and a condition to trigger batch_size incrementing
    condition = stopping_methods.StabilizingPredictions(windows=4, threshold=0.99)
    manager = stopping_methods.Manager(
        [
            stopping_methods.StabilizingPredictions(windows=2, threshold=0.99),
            stopping_methods.StabilizingPredictions(windows=3, threshold=0.99),
            condition,
        ]
    )

    learn(
        estimator,
        query_strategy,
        batch_size,
        unlabeled_pool=Pool(X_unlabeled_pool, y_unlabeled_pool, oh.container.raw_train_set_path),
        test_set=Pool(X_test, y_test, oh.container.raw_test_set_path),
        stop_set=Pool(X_stop_set, y_stop_set, oh.container.raw_stop_set_path),
        stopping=None,  # Stopping(manager, condition, oh.container.stopping_results_csv_file),
    )


if __name__ == "__main__":

    from active_learning import local

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(local.experiment_parameters)
