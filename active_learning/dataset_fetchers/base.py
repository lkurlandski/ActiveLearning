"""Base classes for fetching datasets from several different sources.
"""

from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Generator, Iterable, Tuple

import numpy as np


class DatasetFetcher(ABC):
    """Retrieve the data required for learning."""

    def __init__(self, random_state: int) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed.
        """

        self.random_state = random_state

    @abstractmethod
    def fetch(
        self,
    ) -> Tuple[Iterable[Any], Iterable[Any], Iterable[Any], Iterable[Any], np.ndarray]:
        """Retrieve the data in-memory.

        Returns
        -------
        Tuple[Iterable[Any], Iterable[Any], Iterable[Any], Iterable[Any], Iterable[Any]]
            Train data, test data, train labels, test labels, and target names.
        """
        ...

    @abstractmethod
    def stream(
        self,
    ) -> Tuple[
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Retrieve the data using a memory-efficient streaming approach.

        Returns
        -------
        Tuple[
                Generator[Any, None, None],
                Generator[Any, None, None],
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
            Train data, test data, train labels, test labels, and target names.
        """
        ...
