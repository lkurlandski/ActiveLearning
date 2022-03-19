"""Get training and test data.

TODO
----
-

FIXME
-----
-
"""

from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Tuple

import numpy as np


class DatasetFetcher(ABC):
    """Retrieve the data required for learning."""

    def __init__(self, random_state: int) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        """

        self.random_state = random_state

    @abstractmethod
    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """Retrieve the data in memory.

        Returns
        -------
        Tuple[np.array, np.array, np.array, np.array, np.array]
            train data, test data, train labels, test labels, and target names
        """
        ...

    @abstractmethod
    def stream(self) -> Tuple[Any, Any, Any, Any, np.array]:
        """Retrieve the data using a memory-efficient streaming approach.

        Returns
        -------
        Tuple[Any, Any, Any, Any, np.array]
            train data, test data, train labels, test labels, and target names
        """
        ...
