"""Query strategies for active learning.
"""

from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable, Union

import numpy as np
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from scipy import sparse
from sklearn.base import BaseEstimator


from active_learning import stat_helper


valid_query_strategies = {
    "uncertainty_batch_sampling",
    "entropy_sampling",
    "margin_sampling",
    "uncertainty_sampling",
    "random_sampling",
    "closest_to_hyperplane",
}


def random_sampling(
    learner: BaseEstimator,  # pylint: disable=unused-argument
    X: Union[np.ndarray, sparse.csr_matrix],
    n_instances: int = 1,
) -> np.ndarray:
    """Select random instances for labeling.

    Parameters
    ----------
    learner : BaseEstimator
        A scikit-learn estimator that has a `decision_function` method.
    X : Union[np.ndarray, sparse.csr_matrix]
        The data to be queried.
    n_instances : int, optional
        Number of instances to label, by default 1

    Returns
    -------
    np.ndarray
        The indices of the instances to label.
    """

    query_idx = np.random.choice(X.shape[0], n_instances, replace=False)
    return query_idx, X[query_idx]


def closest_to_hyperplane(
    learner: Union[BaseEstimator, ActiveLearner],
    X: Union[np.ndarray, sparse.csr_matrix],
    n_instances: int = 1,
) -> np.ndarray:
    """Select instances closest the learned model's hyperplane for labeling.

    In the multiclass case, when multiple binary classifiers are ensembled to produced a multiclass
        classifier, the instances with the smallest mean distance to the hyperplane are selected.

    Parameters
    ----------
    learner : Union[BaseEstimator, ActiveLearner]
        A scikit-learn estimator that has a `decision_function` method.
    X : Union[np.ndarray, sparse.csr_matrix]
        The data to be queried.
    n_instances : int, optional
        Number of instances to label, by default 1

    Returns
    -------
    np.ndarray
        The indices of the instances to label.
    """

    if isinstance(learner, ActiveLearner):
        learner = learner.estimator

    y = np.abs(learner.decision_function(X))
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    query_idx = stat_helper.multi_argmin(np.abs(y), n_instances)

    return query_idx, X[query_idx]


def get_modAL_query_strategy(
    query_strategy: str,
) -> Callable[[BaseEstimator, Union[np.ndarray, sparse.csr_matrix]], np.ndarray]:
    """Retrieve the modAL query strategy for active learning.

    Parameters
    ----------
    query_strategy : str
        A string referring to one of the valid query strategies.

    Returns
    -------
    Callable[[BaseEstimator, Union[np.ndarray, sparse.csr_matrix]], np.ndarray]
        A function that takes a scikit-learn estimator and a data matrix and
            returns the indices of the instances to label.

    Raises
    ------
    ValueError
        If the query strategy is not recognized.
    """

    if query_strategy == "uncertrainty_batch_sampling":
        return uncertainty_batch_sampling
    elif query_strategy == "entropy_sampling":
        return entropy_sampling
    elif query_strategy == "margin_sampling":
        return margin_sampling
    elif query_strategy == "uncertainty_sampling":
        return uncertainty_sampling
    elif query_strategy == "random_sampling":
        return random_sampling
    elif query_strategy == "closest_to_hyperplane":
        return closest_to_hyperplane
    else:
        raise ValueError(
            f"query_strategy: {query_strategy} not recognized. "
            f"Valid strategies are:\n{pformat(valid_query_strategies)}"
        )
