"""Extract features from complex data objects, such as text documents.

TODO
----
-

FIXME
-----
-
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import types
from typing import Any, Iterable, Tuple

import numpy as np

from active_learning.feature_extractors.base import FeatureExtractor


class PreprocessedFeatureExtractor(FeatureExtractor):
    """Feature extractor for datasets which are already preprocessed and require no extraction."""

    def extract_features(
        self, X_train: Iterable[Any], X_test: Iterable[Any]
    ) -> Tuple[Iterable[Any], Iterable[Any]]:
        """Return the input data object; convert input generators into array type.

        Parameters
        ----------
        X_train : Iterable[Any]
            Training data
        X_test : Iterable[Any]
            Testing data

        Returns
        -------
        Tuple[Iterable[Any], Iterable[Any]]
            Two dimensional feature representations of the input train and test data
        """

        X_train = np.array(list(X_train)) if isinstance(X_train, types.GeneratorType) else X_train
        X_test = np.array(list(X_test)) if isinstance(X_test, types.GeneratorType) else X_test

        return X_train, X_test
