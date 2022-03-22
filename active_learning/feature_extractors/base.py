"""Extract features from complex data objects, such as text documents.

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
from typing import Any, Iterable, Tuple


class FeatureExtractor(ABC):
    """Tool for extracting features from complex data objects."""

    @abstractmethod
    def extract_features(
        self, X_train: Iterable[Any], X_test: Iterable[Any]
    ) -> Tuple[Iterable[Any], Iterable[Any]]:
        """Extract the features from a dataset.

        Parameters
        ----------
        X_train : Any
            Training data
        X_test : Any
            Testing data

        Returns
        -------
        Tuple[Any, Any]
            Two dimensional feature representations of the input train and test data
        """
