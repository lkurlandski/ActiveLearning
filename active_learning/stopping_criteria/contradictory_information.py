"""
"""

from __future__ import annotations
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from active_learning.stopping_criteria.base import StoppingCriteria


class ContradictoryInformation(StoppingCriteria):

    windows: int
    stop_set_size: Union[float, int]
    stop_set_indices: np.ndarray
    conf_scores: List[float]

    def __init__(
        self,
        windows: int,
        mode: str,
        stop_set_size: Union[float, int],
    ) -> None:

        super().__init__()

        self.windows = windows
        self.mode = mode
        self.stop_set_size = stop_set_size
        self.conf_scores = []
        self.stop_set_indices = None

    def get_hyperparams(self) -> Dict[str, float]:

        return {
            "windows": self.windows,
            "mode": self.mode,
            "stop_set_size": self.stop_set_size,
        }

    def update_from_confs(
        self,
        *,
        stop_set_confs: Optional[np.ndarray] = None,
        initial_unlabeled_pool_confs: Optional[np.ndarray] = None,
    ) -> ContradictoryInformation:

        # Handle improper arguments
        if stop_set_confs is None and initial_unlabeled_pool_confs is None:
            raise ValueError()
        if stop_set_confs is not None and initial_unlabeled_pool_confs is not None:
            raise ValueError()

        # Convert to np arrays if lists were supplied
        if isinstance(stop_set_confs, list):
            stop_set_confs = np.array(stop_set_confs)
        if isinstance(initial_unlabeled_pool_confs, list):
            initial_unlabeled_pool_confs = np.array(initial_unlabeled_pool_confs)

        # Acquire the predictions on the stop set, confs
        if stop_set_confs is None:
            if self.stop_set_indices is None:
                self.set_stop_set_indices(initial_unlabeled_pool_confs.shape[0])
            stop_set_confs = initial_unlabeled_pool_confs[self.stop_set_indices]

        # Check dimensions of input
        if stop_set_confs.ndim > 2:
            raise ValueError()

        # Determine the confidence score
        confidence = np.mean(stop_set_confs)

        # Update the agreement scores and set the has stopped parameter
        self.conf_scores.append(confidence)
        self.set_has_stopped()

        return self

    def update_from_model(
        self,
        model: Callable[..., np.ndarray],
        predict_conf: Callable[..., np.ndarray],
        *,
        stop_set: Optional[np.ndarray] = None,
        initial_unlabeled_pool: Optional[np.ndarray] = None,
    ) -> ContradictoryInformation:

        if stop_set is None and initial_unlabeled_pool is None:
            raise ValueError()
        if stop_set is not None and initial_unlabeled_pool is not None:
            raise ValueError()

        if stop_set is None:
            stop_set = initial_unlabeled_pool[0 : self.stop_set_size]
        stop_set_confs = predict_conf(model, stop_set)
        return self.update_from_confs(stop_set_confs=stop_set_confs)

    def set_stop_set_indices(self, initial_unlabaled_pool_size):

        if self.stop_set_size is None:
            self.stop_set_size = "?"
            return self

        size = float(self.stop_set_size)
        size = int(size) if size.is_integer() else int(size * initial_unlabaled_pool_size)
        size = min(size, initial_unlabaled_pool_size)
        self.stop_set_indices = np.random.choice(initial_unlabaled_pool_size, size, replace=False)

        return self

    def set_has_stopped(self) -> ContradictoryInformation:

        if len(self.conf_scores) + 1 < self.windows:
            self.has_stopped = False
            return self

        elif self.mode == "decreasing":
            prv_conf = self.conf_scores[-self.windows - 1]
            for conf in self.conf_scores[-self.windows:]:
                if conf >= prv_conf:
                    return self
            self.has_stopped = True

        elif self.mode == "nonincreasing":
            prv_conf = self.conf_scores[-self.windows - 1]
            for conf in self.conf_scores[-self.windows:]:
                if conf >= prv_conf:
                    return self
                prv_conf = conf
            self.has_stopped = True

        return self
