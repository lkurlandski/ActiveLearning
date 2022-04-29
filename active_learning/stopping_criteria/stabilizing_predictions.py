"""Stopping methods based upon stabilizing predictions.
"""

from __future__ import annotations
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable, Dict, List, Tuple, Optional, Union
import warnings

from nltk.metrics import masi_distance, binary_distance
from nltk.metrics.agreement import AnnotationTask
import numpy as np
from scipy import sparse
from sklearn.utils.multiclass import type_of_target

from active_learning.stopping_criteria.base import StoppingCriteria, get_stop_set_indices


def get_agreement_metric(preds: np.ndarray) -> str:
    """Determine the agreement metric to use for a particular type of target.

    Parameters
    ----------
    preds : np.ndarray
        The predictions that will be used to compute agreement upon.

    Returns
    -------
    str
        A string identifier corresponding to one of the agreement metrics in nltk.metrics.agreement.

    Raises
    ------
    ValueError
        If the type of target is not supported.
    """

    target_type = type_of_target(preds)

    if target_type == "binary":
        return "kappa"
    if target_type == "multiclass":
        return "kappa"
    if target_type == "multilabel-indicator":
        return "alpha"

    raise ValueError(f"Unsupported target type: {target_type}")


class StabilizingPredictions(StoppingCriteria):
    """Stabilizing Predictions stopping method.

    Attributes
    ----------
    windows : int
        The number of previous agreement scores to consider when determining whether to stop.
    threshold : float
        The threshold that the mean agreement must exceed to stop.
    stop_set_size : Union[float, int]
        The size of the stop set. This is only used if the stop set is not supplied as an argument
            to one of the update methods.
    agreement_metric : str
        An agreement metric to use when determining agreement between consecutive models.
    stop_set_indices : Optional[np.ndarray]
        The indices of the stop set in the initial unlabeled pool.
    prv_stop_set_preds : np.ndarray
        The previous model's predictions on the stop set.
    agreement_scores : List[float]
        The list agreement scores since the beginning of AL.
    """

    windows: int
    threshold: float
    agreement_metric: str
    stop_set_size: Union[float, int]
    stop_set_indices: np.ndarray
    prv_stop_set_preds: np.ndarray
    agreement_scores: List[float]

    def __init__(
        self,
        windows: int,
        threshold: float,
        stop_set_size: Union[float, int],
        agreement_metric: Optional[str] = None,
    ) -> None:
        """Instantiate an instance of the StabilizingPredictions class.

        Parameters
        ----------
        windows : int
            The number of previous agreement scores to consider when determining whether to stop.
        threshold : float
            The threshold that the mean agreement must exceed to stop.
        stop_set_size : Union[float, int]
            The size of the stop set.
        agreement_metric : Optional[str]
            An agreement metric to use when determining agreement between consecutive models.
                If None, the agreement metric defaults to kappa in the binary/multiclass case and
                alpha in the multilabel case.
        """

        super().__init__()

        self.windows = windows
        self.threshold = threshold
        self.stop_set_size = stop_set_size
        self.agreement_metric = agreement_metric
        self.agreement_scores = []
        self.prv_stop_set_preds = None
        self.stop_set_indices = None

    def get_hyperparams(self) -> Dict[str, float]:

        return {
            "threshold": self.threshold,
            "windows": self.windows,
            "stop_set_size": self.stop_set_size,
            "agreement_metric": self.agreement_metric,
        }

    def update_from_preds(
        self,
        *,
        stop_set_preds: Optional[np.ndarray] = None,
        initial_unlabeled_pool_preds: Optional[np.ndarray] = None,
    ) -> StabilizingPredictions:
        """Update the stopping method from pre-computed predictions.

        Parameters
        ----------
        stop_set_preds : Optional[np.ndarray], optional
            Predictions on a pre-selected stop set, by default None
        initial_unlabeled_pool_preds : Optional[np.ndarray], optional
            Predictions on the entire unlabeled pool, by default None

        Returns
        -------
        StabilizingPredictions
            An updated instance of the StabilizingPredictions class.

        Raises
        ------
        ValueError
            If both stop_set_preds and initial_unlabeled_pool_preds are None or both are not None.
        """

        if (stop_set_preds is None and initial_unlabeled_pool_preds is None) or (
            stop_set_preds is not None and initial_unlabeled_pool_preds is not None
        ):
            raise ValueError(
                "Only one of stop_set_preds or initial_unlabeled_pool_preds should be supplied."
            )

        # Convert to np arrays if lists were supplied
        if isinstance(stop_set_preds, list):
            stop_set_preds = np.array(stop_set_preds)
        if isinstance(initial_unlabeled_pool_preds, list):
            initial_unlabeled_pool_preds = np.array(initial_unlabeled_pool_preds)
        
        # TODO: add support for sparse matrices instead of casting to dense
        if isinstance(stop_set_preds, sparse.csr_matrix):
            stop_set_preds = stop_set_preds.toarray()
        if isinstance(initial_unlabeled_pool_preds, sparse.csr_matrix):
            initial_unlabeled_pool_preds = initial_unlabeled_pool_preds.toarray()

        # Acquire the predictions on the stop set
        if stop_set_preds is None:
            # Set the stop set indices if they have not been set yet
            if self.stop_set_indices is None:
                self.stop_set_indices = get_stop_set_indices(
                    self.stop_set_size, initial_unlabeled_pool_preds.shape[0]
                )
            else:
                self.stop_set_indices = self.stop_set_indices
            # Select the stop set predictions
            stop_set_preds = initial_unlabeled_pool_preds[self.stop_set_indices]

        # Determine the agreement score
        agreement = self.get_agreement(stop_set_preds)

        # Update the agreement scores and set the has stopped parameter
        self.agreement_scores.append(agreement)
        self.prv_stop_set_preds = stop_set_preds
        self.set_has_stopped()

        return self

    def update_from_model(
        self,
        model: Callable[..., np.ndarray],
        predict: Callable[..., np.ndarray],
        *,
        stop_set: Optional[np.ndarray] = None,
        initial_unlabeled_pool: Optional[np.ndarray] = None,
    ) -> StabilizingPredictions:
        """Update the stopping method by computing predictions on the stop set.

        Parameters
        ----------
        model : Callable[..., np.ndarray]
            A model that can be used to make predictions on the unlabeled pool.
        predict : Callable[..., np.ndarray]
            A function that takes a model and a dataset and returns predictions.
        stop_set : Optional[np.ndarray], optional
            A pre-selected stop set, by default None
        initial_unlabeled_pool : Optional[np.ndarray], optional
            The entire unlabeled pool, by default None

        Returns
        -------
        StabilizingPredictions
            An updated instance of the StabilizingPredictions class.

        Raises
        ------
        ValueError
            If both stop_set and initial_unlabeled_pool are None or both are not None.
        """

        if (stop_set is None and initial_unlabeled_pool is None) or (
            stop_set is not None and initial_unlabeled_pool is not None
        ):
            raise ValueError("Only one of stop_set or initial_unlabeled_pool should be supplied.")

        if stop_set is None:
            stop_set = initial_unlabeled_pool[0 : self.stop_set_size]
        stop_set_preds = predict(model, stop_set)
        return self.update_from_preds(stop_set_preds=stop_set_preds)

    def set_has_stopped(self) -> StabilizingPredictions:
        """Update the instances's has_stopped attribute based on the current agreement scores.

        Returns
        -------
        StabilizingPredictions
            Updated instance of self.
        """

        if len(self.agreement_scores[1:]) < self.windows:
            self.has_stopped = False
        elif np.mean(self.agreement_scores[-self.windows :]) > self.threshold:
            self.has_stopped = True
        else:
            self.has_stopped = False

        return self

    def get_agreement(self, stop_set_preds: np.ndarray) -> float:
        """Get the agreement between the previous and current stop set predictions.

        Parameters
        ----------
        stop_set_preds : np.ndarray
            The current stop set predictions.

        Returns
        -------
        float
            The agreement score.

        Raises
        ------
        ValueError
            If the agreement metric is unknown by the system.
        """

        def data(multilabel: bool) -> List[Tuple[int, int, Union[int, frozenset]]]:
            """Extract data compatible with nltk's AnnotationTask from stop set predictions.

            Arguments
            ---------
            multilabel: bool
                If True, assumes the stop_set_predictions are a two dimensional multilabel array,
                    finds the positive classes in the array, and adds them to a frozenset.

            Returns
            -------
            List[Tuple[int, int, Union[int, frozenset]]]
                A list of tuples, where each tuple is of the form (coder, item, label(s)).
            """

            if multilabel:
                dat = [
                    (0, i, frozenset(np.flatnonzero(l))) for i, l in enumerate(stop_set_preds)
                ] + [
                    (1, i, frozenset(np.flatnonzero(l)))
                    for i, l in enumerate(self.prv_stop_set_preds)
                ]
            else:
                dat = [(0, i, l) for i, l in enumerate(stop_set_preds)] + [
                    (1, i, l) for i, l in enumerate(self.prv_stop_set_preds)
                ]

            return dat

        # Set the agreement metric if it hasn't been set yet
        self.agreement_metric = (
            get_agreement_metric(stop_set_preds)
            if self.agreement_metric is None
            else self.agreement_metric
        )

        if self.prv_stop_set_preds is None:
            return np.NaN
        if np.array_equal(self.prv_stop_set_preds, stop_set_preds):
            return 1.0

        try:
            if self.agreement_metric == "kappa":
                return AnnotationTask(data(False), binary_distance).kappa()
            if self.agreement_metric == "S":
                return AnnotationTask(data(False), binary_distance).S()
            if self.agreement_metric == "pi":
                return AnnotationTask(data(True), masi_distance).pi()
            if self.agreement_metric == "alpha":
                return AnnotationTask(data(True), masi_distance).alpha()
        except ZeroDivisionError:
            warnings.warn(
                "Encountered a ZeroDivisionError computing agreement with "
                f"{self.agreement_metric}. Returning NaN for now, but someone should investigate."
            )
            return np.NaN

        raise ValueError(f"Unknown agreement metric: {self.agreement_metric}")
