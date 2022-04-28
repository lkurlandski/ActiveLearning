"""Base stopping method abstract class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Dict, Union

import numpy as np


def get_stop_set_indices(size: Union[int, float], initial_unlabaled_pool_size: int) -> np.ndarray:
    """Get the indices for a stop set.

    Parameters
    ----------
    size : Union[int, float]
        The size of the stop set. If a float, it is interpreted as a percentage of the initial
            unlabeled pool size. If an int, it is interpreted as the number of indices to select.
    initial_unlabaled_pool_size : int
        The initial size of the unlabeled pool, from which the stop set is selected.

    Returns
    -------
    np.ndarray
        The indices of the unlabeled pool to use for the stop set.
    """

    size = float(size)
    size = int(size) if size.is_integer() else int(size * initial_unlabaled_pool_size)
    size = min(size, initial_unlabaled_pool_size)
    idx = np.random.choice(initial_unlabaled_pool_size, size, replace=False)

    return idx


class StoppingCriteria(ABC):
    """Base class for stopping criteria.

    Attributes
    ----------
    has_stopped : bool
        Boolean indicating if the stopping criteria has stopped.
    """

    def __init__(self):
        """Instantiate a stopping criteria."""

        self.has_stopped = False

    def __str__(self):
        return (
            type(self).__name__
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.get_hyperparams().items()])
            + ")"
        )

    @abstractmethod
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return a dictionary of hyperparameters specific to a subclass of StoppingCriteria.

        Returns
        -------
        Dict[str, Any]
            Hyperparamters, used in the __str__ method.
        """
        ...
