"""Run the active learning process.

TODO
----
- Implement a label-encoding strategy for the target array.
- Introduce checks to ensure the base learner can accept sparse arrays and such.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import

import datetime
from pathlib import Path
import random
import time
from typing import Callable, Union
import warnings

import joblib
from modAL.models import ActiveLearner
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import unique_labels, type_of_target

from active_learning import estimators
from active_learning import stat_helper
from active_learning import dataset_fetchers
from active_learning import feature_extractors
from active_learning import query_strategies
from active_learning import utils
from active_learning.active_learners.helpers import Params, Pool, OutputHelper
from active_learning.stopping_criteria import stabilizing_predictions


def update(
    start_time: float,
    unlabeled_init_size: int,
    unlabeled_size: int,
    batch_size: int,
    i: int,
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
    batch_size : int
        The number of examples selected at each iteration for labeling.
    i : int
        The current iteration of AL
    accuracy : float
        The current accuracy of the model on a test set
    """

    prop = (unlabeled_init_size - unlabeled_size) / unlabeled_init_size

    pool_zfill = len(str(unlabeled_init_size))

    print(
        f"itr: {str(i).zfill(4)}"
        f"    pgs: {str(round(100 * prop, 2)).zfill(5)}%"
        f"    time: {str(datetime.timedelta(seconds=(round(time.time() - start_time))))}"
        f"    |U_0|: {unlabeled_init_size}"
        f"    |U|: {str(unlabeled_size).zfill(pool_zfill)}"
        f"    |L|: {str(unlabeled_init_size - unlabeled_size).zfill(pool_zfill)}",
        f"    |B|: {batch_size}",
        flush=True,
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


def get_batch_size(
    batch_size: int, unlabeled_pool_size: int, early_stop_mode_triggered: bool, early_stop_mode: str
) -> int:
    """Gets the batch size at the current iteration of AL.

    Parameters
    ----------
    batch_size : int
        The previous batch size.
    unlabeled_pool_size : int
        The current size of the unlabeled pool.
    early_stop_mode_triggered : bool
        Whether or not the optional early stopping mode has been triggered.
    early_stop_mode : str
        The mode of early stopping. One of {"exponential", "finish", "none"}.

    Returns
    -------
    int
        The batch size to use at the current iteration of AL.
    """

    if early_stop_mode_triggered:
        if early_stop_mode == "exponential":
            batch_size = batch_size * 2
        elif early_stop_mode == "finish":
            batch_size = unlabeled_pool_size

    batch_size = unlabeled_pool_size if batch_size > unlabeled_pool_size else batch_size

    return batch_size


def adjust_first_batch_if_nessecary(
    idx: np.ndarray,
    estimator: BaseEstimator,
    y_unlabeled_pool: Union[np.ndarray, sparse.csr_matrix],
    batch_size: int,
    first_batch_mode: str,
) -> np.ndarray:
    """Ammend the first batch to correct for a scikit-learn bug in the OneVsRestClassifier.

    Scikit-learn OneVsRestClassifier has a bug that causes a division error when only one class
        is present in the training data. This is a reasonable workaround that will repeatedly select
        a random batch until the batch contains at least two distinct classes. You can read about
        the bug at this issue: https://github.com/scikit-learn/scikit-learn/issues/21869. The bug
        only appears under a particular set of circumstances.

    Parameters
    ----------
    estimator : BaseEstimator
        Scikit-learn estimator.
    y_unlabeled_pool : Union[np.ndarray, sparse.csr_matrix]
        The unlabeled pool, used to determine the type of target.
    batch_size : int
        Desired initial batch size.
    first_batch_mode : str
        The mode of the first batch.

    Returns
    -------
    np.ndarray
        The batch indices for the 0th iteration of AL.
    """

    if (
        isinstance(estimator, OneVsRestClassifier)
        and type_of_target(y_unlabeled_pool) == "multiclass"
        and len(set(y_unlabeled_pool[idx])) == 1
    ):

        warnings.warn("Implementing a solution for a bug from scikit-learn.")

        while len(set(y_unlabeled_pool[idx])) == 1:
            idx = get_first_batch(y_unlabeled_pool, protocol=first_batch_mode, k=max(batch_size, 2))

    return idx


def learn(
    estimator: BaseEstimator,
    query_strategy: Callable,
    batch_size: int,
    unlabeled_pool: Pool,
    *,
    model_path: Path = None,
    batch_path: Path = None,
    test_set: Pool = None,
    first_batch_mode: str = "random",
    early_stop_mode: str = "none",
):
    """Perform active learning and save a limited amount of information to files.

    Parameters
    ----------
    estimator : BaseEstimator
        A modAL compatible estimator to learn.
    query_strategy : Callable
        A modAL compatible query strategy.
    batch_size : int
        The number of examples to select for labeling at each iteration.
    unlabeled_pool : Pool
        The initial unlabeled pool of training data.
    model_path : Path, optional
        A location to save trained models to, by default None, which means no models are saved.
    batch_path : Path, optional
        A location to save the examples queried for labeling to, by default None, which means no
            batch files are saved.
    test_set : Pool, optional
        A test set, which will be saved in a compressed format, by default None, which means no
            test set is saved.
    first_batch_mode : str, optional
        Determines how the first batch is selected, by default "random".
    early_stop_mode : str, optional
        Determines how the batch size is adjusted, by default "none".
    """
    print("-" * 80, flush=True)
    start = time.time()
    print("0:00:00 -- Starting Active Learning", flush=True)

    # Save the sets to perform inference downstream
    unlabeled_pool.save()
    if test_set is not None:
        test_set.save()

    unlabeled_init_size = unlabeled_pool.y.shape[0]
    i = 0

    # Set up the Stabilizing Predictions stopping method, if requested
    early_stop_mode_triggered = False
    stopper, initial_unlabeled_pool = None, None
    if early_stop_mode != "none":
        stopper = stabilizing_predictions.StabilizingPredictions(
            windows=3, threshold=0.99, stop_set_size=1000
        )
        initial_unlabeled_pool = unlabeled_pool.X.copy()

    # Perform the 0th iteration of active learning
    idx = get_first_batch(unlabeled_pool.y, protocol=first_batch_mode, k=batch_size)
    idx = adjust_first_batch_if_nessecary(
        idx, estimator, unlabeled_pool.y, batch_size, first_batch_mode
    )
    learner = ActiveLearner(estimator=estimator, query_strategy=query_strategy)
    learner.teach(unlabeled_pool.X[idx], unlabeled_pool.y[idx])

    # Begin active learning loop
    while unlabeled_pool.y.shape[0] > 0:

        # Retrain the learner during every iteration, except the 0th one
        if i > 0:
            batch_size = get_batch_size(
                batch_size,
                unlabeled_pool.y.shape[0],
                early_stop_mode_triggered,
                early_stop_mode,
            )
            idx, query_sample = learner.query(unlabeled_pool.X, n_instances=batch_size)
            query_labels = unlabeled_pool.y[idx]
            learner.teach(query_sample, query_labels)

        if model_path is not None:
            joblib.dump(learner.estimator, model_path / f"{i}.joblib")
        if batch_path is not None:
            np.save(batch_path / f"{i}.npy", idx)

        # Remove the queried elements from the unlabeled pool
        unlabeled_pool.X = stat_helper.remove_ids_from_array(unlabeled_pool.X, idx)
        unlabeled_pool.y = stat_helper.remove_ids_from_array(unlabeled_pool.y, idx)

        # Perform end-of-iteration tasks
        update(start, unlabeled_init_size, unlabeled_pool.y.shape[0], batch_size, i)
        i += 1

        if early_stop_mode != "none" and not early_stop_mode_triggered:
            stopper.update_from_model(
                model=learner,
                predict=lambda model, X: model.predict(X),
                initial_unlabeled_pool=initial_unlabeled_pool,
            )
            if stopper.has_stopped:
                print(f"Early stoppping triggered at iteration {i} with mode: {early_stop_mode}")
                early_stop_mode_triggered = True

    # End active learning loop
    diff = datetime.timedelta(seconds=(round(time.time() - start)))
    print(f"{diff} -- Ending Active Learning", flush=True)
    print("-" * 80, flush=True)


def main(params: Params) -> None:
    """Run the active learning algorithm for a set of experiment parmameters.

    Parameters
    ----------
    params : Params
        Parameters for the experiment.
    """

    # Seed random, numpy.random, and get a numpy randomizer
    random.seed(params.random_state)
    np.random.seed(params.random_state)

    # Get the dataset and perform the feature extraction
    (
        X_unlabeled_pool,
        X_test,
        y_unlabeled_pool,
        y_test,
        _,
    ) = dataset_fetchers.get_dataset(params.dataset, stream=True, random_state=params.random_state)
    X_unlabeled_pool, X_test = feature_extractors.get_features(
        X_unlabeled_pool, X_test, params.feature_rep
    )
    unlabeled_init_size = y_unlabeled_pool.shape[0]

    # Get the batch size (handle proportions and absolute values)
    if isinstance(params.batch_size, float):
        batch_size = int(params.batch_size * unlabeled_init_size)
    else:
        batch_size = params.batch_size
    batch_size = max(1, batch_size)

    # Get the modAL compatible estimator and query strategy to use
    estimator = estimators.get_estimator(
        params.base_learner,
        type_of_target(y_unlabeled_pool),
        random_state=params.random_state,
    )
    query_strategy = query_strategies.get_modAL_query_strategy(params.query_strategy)

    # Setup output directory structure
    oh = OutputHelper(params)
    oh.setup()
    oh.container.set_target_vector_ext(utils.get_array_file_ext(y_unlabeled_pool))

    unlabeled_pool = Pool(
        X_unlabeled_pool,
        y_unlabeled_pool,
        oh.container.X_unlabeled_pool_file,
        oh.container.y_unlabeled_pool_file,
    )
    test_set = Pool(X_test, y_test, oh.container.X_test_set_file, oh.container.y_test_set_file)

    learn(
        estimator,
        query_strategy,
        batch_size,
        unlabeled_pool,
        model_path=oh.container.model_path,
        batch_path=oh.container.batch_path,
        test_set=test_set,
        first_batch_mode=params.first_batch_mode,
        early_stop_mode=params.early_stop_mode,
    )
