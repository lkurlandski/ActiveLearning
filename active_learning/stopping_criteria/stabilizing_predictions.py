"""
"""

from __future__ import annotations
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.multiclass import type_of_target

from active_learning.stopping_criteria.base import StoppingCriteria


def determine_agreement_metric(preds):

    target_type = type_of_target(preds)
    if target_type == "binary" or target_type == "multiclass":
        return cohen_kappa_score
    elif target_type == "multilabel-indicator":
        raise NotImplementedError()
    raise ValueError(f"Target type: {target_type} not supported")


class StabilizingPredictions(StoppingCriteria):
    """Stablizing Predictions stopping method."""

    agreement_scores: List[float]
    prv_preds: np.ndarray
    preds: np.ndarray

    def __init__(
        self,
        windows:int,
        threshold:float,
        agreement_metric:Optional[Callable]=None,
        *,
        initial_unlabeled_pool:Optional[Union[np.ndarray, sparse.csr_matrix]]=None,
        stop_set_size:Optional[Union[float, int]]=None
    ) -> None:

        self.windows = windows
        self.threshold = threshold
        self.agreement_metric = agreement_metric
        self.initial_unlabeled_pool = initial_unlabeled_pool
        self.stop_set_size = stop_set_size
        self.stop_set_indices = self.get_stop_set_indices(self.stop_set_size)
        self.agreement_scores = []
        self.prv_preds = None

    def get_stop_set_indices(self, size: Union[int, float]):

        if size is None:
            return None

        size = float(size)
        size = int(size) if size.is_integer() else int(size * self.initial_unlabeled_pool.shape[0])
        size = max(1, size)
        size = min(size, self.initial_unlabeled_pool.shape[0])
        idx = np.random.choice(self.initial_unlabeled_pool.shape[0], size=size, replace=False)
        
        return idx

    def get_hyperparams(self) -> Dict[str, float]:

        return {
            "threshold": self.threshold,
            "windows": self.windows,
            "agreement_metric" : self.agreement_metric
        }

    def update(
        self,
        *,
        model:Callable = None,
        predict:Callable = None,
        preds:np.ndarray = None,
        **kwargs
    ) -> StabilizingPredictions:
        """
            >>> predict = lamnda x, y : x.predict(y)
            >>> s.update(s, status, model, stop_set, predict)

            >>> # Using a pytorch model
            >>> s = ()
            >>> predict x, y : x(y)
        """

        if preds is None and any(x is None for x in (model, predict, self.initial_unlabeled_pool)):
            raise Exception()
        if preds is not None and any(x is not None for x in (model, predict)):
            raise Exception()

        if preds is None:
            preds = predict(model, self.initial_unlabeled_pool[self.stop_set_indices])

        if self.agreement_metric is None:
            self.agreement_metric = determine_agreement_metric(preds)

        if self.prv_preds is None:
            agreement = np.NaN
        elif np.array_equal(self.prv_preds, preds):
            agreement = 1.0
        else:
            agreement = self.agreement_metric(self.prv_preds, preds)

        self.agreement_scores.append(agreement)
        self.prv_preds = preds

        return self

    def is_stop(self) -> bool:
        
        if len(self.agreement_scores[1:]) < self.windows:
            return False

        if np.mean(self.agreement_scores[-self.windows:]) > self.threshold:
            return True

        return False
