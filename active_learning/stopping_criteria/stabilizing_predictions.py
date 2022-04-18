"""
"""

from __future__ import annotations
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable, Dict, List, Tuple, Optional, Union

from nltk.metrics import masi_distance, binary_distance
from nltk.metrics.agreement import AnnotationTask
import numpy as np
from sklearn.utils.multiclass import type_of_target

from active_learning.stopping_criteria.base import StoppingCriteria


class StabilizingPredictions(StoppingCriteria):

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

        # Handle improper arguments
        if stop_set_preds is None and initial_unlabeled_pool_preds is None:
            raise ValueError()
        if stop_set_preds is not None and initial_unlabeled_pool_preds is not None:
            raise ValueError()

        # Convert to np arrays if lists were supplied
        if isinstance(stop_set_preds, list):
            stop_set_preds = np.array(stop_set_preds)
        if isinstance(initial_unlabeled_pool_preds, list):
            initial_unlabeled_pool_preds = np.array(initial_unlabeled_pool_preds)

        # Acquire the predictions on the stop set, preds
        if stop_set_preds is None:
            if self.stop_set_indices is None:
                self.set_stop_set_indices(initial_unlabeled_pool_preds.shape[0])
            stop_set_preds = initial_unlabeled_pool_preds[self.stop_set_indices]

        # Check dimensions of input
        if stop_set_preds.ndim > 2:
            raise ValueError()

        # Determine the agreement metric, if None exists, and the agreement score
        if self.agreement_metric is None:
            self.set_agreement_metric(stop_set_preds)
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

        if stop_set is None and initial_unlabeled_pool is None:
            raise ValueError()
        if stop_set is not None and initial_unlabeled_pool is not None:
            raise ValueError()

        if stop_set is None:
            stop_set = initial_unlabeled_pool[0 : self.stop_set_size]
        stop_set_preds = predict(model, stop_set)
        return self.update_from_preds(stop_set_preds=stop_set_preds)

    def set_stop_set_indices(self, initial_unlabaled_pool_size):

        if self.stop_set_size is None:
            self.stop_set_size = "?"
            return self

        size = float(self.stop_set_size)
        size = int(size) if size.is_integer() else int(size * initial_unlabaled_pool_size)
        size = min(size, initial_unlabaled_pool_size)
        self.stop_set_indices = np.random.choice(initial_unlabaled_pool_size, size, replace=False)

        return self

    def set_has_stopped(self) -> StabilizingPredictions:

        if len(self.agreement_scores[1:]) < self.windows:
            self.has_stopped = False
        elif np.mean(self.agreement_scores[-self.windows :]) > self.threshold:
            self.has_stopped = True
        else:
            self.has_stopped = False

        return self

    def set_agreement_metric(self, preds) -> StabilizingPredictions:

        target_type = type_of_target(preds)

        if target_type == "binary":
            self.agreement_metric = "kappa"
        elif target_type == "multiclass":
            self.agreement_metric = "kappa"
        elif target_type == "multilabel-indicator":
            self.agreement_metric = "alpha"
        else:
            raise ValueError()

        return self

    def get_agreement(self, stop_set_preds):
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

        if self.prv_stop_set_preds is None:
            return np.NaN
        if np.array_equal(self.prv_stop_set_preds, stop_set_preds):
            return 1.0

        if self.agreement_metric == "kappa":
            return AnnotationTask(data(False), binary_distance).kappa()
        elif self.agreement_metric == "S":
            return AnnotationTask(data(False), binary_distance).S()
        elif self.agreement_metric == "pi":
            return AnnotationTask(data(True), masi_distance).pi()
        elif self.agreement_metric == "alpha":
            return AnnotationTask(data(True), masi_distance).alpha()
        else:
            raise ValueError()
