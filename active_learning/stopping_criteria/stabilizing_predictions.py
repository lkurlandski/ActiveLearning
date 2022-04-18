"""
"""

from __future__ import annotations
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable, Dict, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.multiclass import type_of_target

from active_learning.stopping_criteria.base import StoppingCriteria


class StabilizingPredictions(StoppingCriteria):
    """Stablizing Predictions stopping method."""

    def __init__(
        self,
        windows:int,
        threshold:float,
        stop_set_size:Optional[Union[float, int]]=None,
        agreement_metric:Optional[Callable]=None,
    ) -> None:

        super().__init__()

        self.windows = windows
        self.threshold = threshold
        self.agreement_metric = agreement_metric
        self.stop_set_size = stop_set_size

        self.agreement_scores = []
        self.prv_stop_set_preds = None
        self.stop_set_indices = None

    def get_hyperparams(self) -> Dict[str, float]:

        return {
            "threshold": self.threshold,
            "windows": self.windows,
            "stop_set_size": self.stop_set_size,
            "agreement_metric" : self.agreement_metric
        }

    def update_from_preds(
        self,
        *,
        stop_set_preds:Optional[Union[np.ndarray, sparse.csr_matrix]] = None,
        initial_unlabeled_pool_preds:Optional[Union[np.ndarray, sparse.csr_matrix]] = None,
    ) -> StabilizingPredictions:
        
        # Handle improper arguments
        if stop_set_preds is None and initial_unlabeled_pool_preds is None:
            raise ValueError()
        if stop_set_preds is not None and initial_unlabeled_pool_preds is not None:
            raise ValueError()
        
        # Acquire the predictions on the stop set, preds
        if stop_set_preds is None:
            if self.stop_set_indices is None:
                self.set_stop_set_indices(initial_unlabeled_pool_preds.shape[0])
            stop_set_preds = initial_unlabeled_pool_preds[self.stop_set_indices]

        # Determine the agreement metric, if None exists
        if self.agreement_metric is None:
            self.set_agreement_metric(stop_set_preds)

        # Determine the agreement score
        if self.prv_stop_set_preds is None:
            agreement = np.NaN
        elif np.array_equal(self.prv_stop_set_preds, stop_set_preds):
            agreement = 1.0
        else:
            agreement = self.agreement_metric(self.prv_stop_set_preds, stop_set_preds)

        # Update the agreement scores and set the has stopped parameter
        self.agreement_scores.append(agreement)
        self.prv_stop_set_preds = stop_set_preds
        self.set_has_stopped()

        return self

    def update_from_model_and_predict(
        self,
        model:Callable[..., Union[np.ndarray, sparse.csr_matrix]],
        predict:Callable[..., Union[np.ndarray, sparse.csr_matrix]],
        *,
        stop_set:Optional[Union[np.ndarray, sparse.csr_matrix]]=None,
        initial_unlabeled_pool:Optional[Union[np.ndarray, sparse.csr_matrix]]=None
    ) -> StabilizingPredictions:

        if stop_set is None and initial_unlabeled_pool is None:
            raise ValueError()
        if stop_set is not None and initial_unlabeled_pool is not None:
            raise ValueError()

        if stop_set is None:
            stop_set = initial_unlabeled_pool[0:self.stop_set_size]
        stop_set_preds = predict(model, stop_set)
        return self.update_from_preds(stop_set_preds=stop_set_preds)

    def set_agreement_metric(self, preds) -> StabilizingPredictions:

        target_type = type_of_target(preds)
        if target_type == "binary" or target_type == "multiclass":
            self.agreement_metric = cohen_kappa_score
        elif target_type == "multilabel-indicator":
            raise NotImplementedError()
        else:
            raise ValueError()

        return self

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
        elif np.mean(self.agreement_scores[-self.windows:]) > self.threshold:
            self.has_stopped = True
        else:
            self.has_stopped = False

        return self
