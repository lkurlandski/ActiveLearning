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
from typing import Any, Tuple


class FeatureExtractor(ABC):
    """Tool for extracting features from complex data objects."""

    def __init__(self, stream: bool) -> None:
        """Create instance.

        Parameters
        ----------
        stream : bool
            Whether or not the input data will be streamed. This may be interpretted differently at
                the subclass level.
        """

        self.stream = stream

    @abstractmethod
    def extract_features(self, X_train: Any, X_test: Any) -> Tuple[Any, Any]:
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
