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
from typing import Any, Iterable, Tuple, Union

import numpy as np
from scipy import sparse


class FeatureExtractor(ABC):
    """Tool for extracting features from complex data objects."""

    @abstractmethod
    def extract_features(
        self, X_train: Iterable[Any], X_test: Iterable[Any]
    ) -> Tuple[Union[np.ndarray, sparse.csr_matrix], Union[np.ndarray, sparse.csr_matrix]]:
        """Extract the features from a dataset.

        Parameters
        ----------
        X_train : Iterable[Any]
            Training data.
        X_test : Iterable[Any]
            Testing data.

        Returns
        -------
        Tuple[Union[np.ndarray, sparse.csr_matrix], Union[np.ndarray, sparse.csr_matrix]]
            Numeric representation of the train and test data.
        """
        ...
