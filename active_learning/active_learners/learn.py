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

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import

from dataclasses import dataclass
import datetime
import joblib
from pathlib import Path
import random
import time
from typing import Callable, Dict, Union

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import numpy as np
from scipy import sparse
from scipy.io import mmwrite, mmread
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels, type_of_target

from active_learning import estimators
from active_learning import stat_helper
from active_learning import dataset_fetchers
from active_learning import feature_extractors
from active_learning.active_learners import output_helper


query_strategies = {
    "entropy_sampling": entropy_sampling,
    "margin_sampling": margin_sampling,
    "uncertainty_batch_sampling": uncertainty_batch_sampling,
    "uncertainty_sampling": uncertainty_sampling,
}


# TODO: implement as a class... with @classmethod for load
@dataclass
class Pool:

    X: Union[np.ndarray, sparse.csr_matrix]
    y: Union[np.ndarray, sparse.csr_matrix]
    X_path: Path
    y_path: Path
    
    def save(self) -> None:
        if self.X_path is None or self.y_path is None:
            raise Exception()
  
        if self.X.ndim == 2 and self.X_path.suffix == ".mtx":
            mmwrite(self.X_path, self.X)
        else:
            raise Exception(f"Extension {self.X_path.name} incompatible with ndim {self.X.ndim}.")

        if self.y.ndim == 2 and self.y_path.suffix == ".mtx":
            mmwrite(self.y_path, self.y)
        elif self.y.ndim == 1 and self.y_path.suffix == ".npy":
            np.save(self.y_path, self.y)
        else:
            raise Exception(f"Extension {self.y_path.name} incompatible with ndim {self.y.ndim}.")

    def load(self) -> None:
        if self.X_path is None or self.y_path is None:
            raise Exception()
        
        if self.X_path.suffix == ".mtx":
            self.X = mmread(self.X_path)
            if sparse.isspmatrix_coo(self.X):
                self.X = sparse.csr_matrix(self.X)
        elif self.X_path.suffix == ".npy":
            self.X = np.load(self.X_path)
        else:
            raise Exception(f"Extension {self.X_path.name} incompatible with known read methods.")

        if self.y_path.suffix == ".mtx":
            self.y = mmread(self.y_path)
            if sparse.isspmatrix_coo(self.y):
                self.y = sparse.csr_matrix(self.y)
        elif self.y_path.suffix == ".npy":
            self.y = np.load(self.y_path)
        else:
            raise Exception(f"Extension {self.y_path.name} incompatible with known read methods.")

        return self

def update(start_time: float, unlabeled_init_size: int, unlabeled_size: int, i: int) -> None:
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

    print(
        f"itr: {str(i).zfill(4)}"
        f"    pgs: {str(round(100 * prop, 2)).zfill(5)}%"
        f"    time: {str(datetime.timedelta(seconds=(round(time.time() - start_time))))}"
        f"    |U_0|: {unlabeled_init_size}"
        f"    |U|: {str(unlabeled_size).zfill(pool_zfill)}"
        f"    |L|: {str(unlabeled_init_size - unlabeled_size).zfill(pool_zfill)}",
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


def learn(
    estimator: BaseEstimator,
    query_strategy: Callable,
    batch_size: int,
    unlabeled_pool: Pool,
    *,
    model_path: Path = None,
    batch_path: Path = None,
    test_set: Pool = None,
    stop_set: Pool = None,
):
    def get_batch_size(batch_size) -> int:
        if batch_size > unlabeled_pool.y.shape[0]:
            batch_size = unlabeled_pool.y.shape[0]
        return batch_size

    # Save the sets to perform inference downstream
    unlabeled_pool.save()
    if test_set is not None:
        test_set.save()
    if stop_set is not None:
        stop_set.save()

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

        if model_path is not None:
            joblib.dump(learner.estimator, model_path / f"{i}.joblib")
        if batch_path is not None:
            np.save(batch_path / f"{i}.npy", idx)

        # Remove the queried elements from the unlabeled pool
        unlabeled_pool.X = stat_helper.remove_ids_from_array(unlabeled_pool.X, idx)
        unlabeled_pool.y = stat_helper.remove_ids_from_array(unlabeled_pool.y, idx)

        # Perform end-of-iteration tasks
        update(start, unlabeled_init_size, unlabeled_pool.y.shape[0], i)
        i += 1

    # End the al experience
    end = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(end - start))))} -- Ending Active Learning")


def main(params: Dict[str, Union[str, int]]) -> None:
    """Run the active learning algorithm for a set of experiment parmameters.

    Parameters
    ----------
    params : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    random_state = int(params["random_state"])

    # Seed random, numpy.random, and get a numpy randomizer
    random.seed(random_state)
    np.random.seed(random_state)
    rng = np.random.default_rng(random_state)

    # Get the dataset and perform the feature extraction
    (
        X_unlabeled_pool,
        X_test,
        y_unlabeled_pool,
        y_test,
        _,
    ) = dataset_fetchers.get_dataset(params["dataset"], stream=True, random_state=random_state)
    X_unlabeled_pool, X_test = feature_extractors.get_features(
        X_unlabeled_pool, X_test, params["feature_representation"]
    )
    unlabeled_init_size = y_unlabeled_pool.shape[0]

    # Get the batch size (handle proportions and absolute values)
    tmp = float(params["batch_size"])
    batch_size = int(tmp) if tmp.is_integer() else int(tmp * unlabeled_init_size)
    batch_size = max(1, batch_size)

    # Get the stop set size (handle proportions and absolute values)
    tmp = float(params["stop_set_size"])
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
        params["base_learner"],
        type_of_target(y_unlabeled_pool),
        random_state=random_state,
    )
    query_strategy = query_strategies[params["query_strategy"]]

    # Setup output directory structure
    oh = output_helper.OutputHelper(params)
    oh.setup()
    oh.container.set_target_vectors(y_unlabeled_pool)
    
    unlabeled_pool = Pool(
        X_unlabeled_pool,
        y_unlabeled_pool,
        oh.container.X_unlabeled_pool_file,
        oh.container.y_unlabeled_pool_file,
    )
    test_set = Pool(
        X_test,
        y_test,
        oh.container.X_test_set_file,
        oh.container.y_test_set_file
    )
    stop_set = Pool(
        X_stop_set,
        y_stop_set,
        oh.container.X_stop_set_file,
        oh.container.y_stop_set_file
    )

    learn(
        estimator,
        query_strategy,
        batch_size,
        unlabeled_pool=unlabeled_pool,
        model_path=oh.container.model_path,
        batch_path=oh.container.batch_path,
        test_set=test_set,
        stop_set=stop_set,
    )
