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
from typing import Callable, Iterable, Tuple, Union

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from active_learning.feature_extractors.base import FeatureExtractor


class ScikitLearnTextFeatureExtractor(FeatureExtractor):
    """Feature extractor using scikit-learn utilities."""

    def __init__(
        self,
        vectorizer: Callable[..., Union[CountVectorizer, HashingVectorizer, TfidfVectorizer]],
        **kwargs,
    ) -> None:
        """Create instance with a custom vectorizer.

        Parameters
        ----------
        stream : bool
            Whether or not the input data will be streamed. In the case of text documents,
                streamed data is expected to be a numpy array of filenames containing raw text. Non-
                streamed data is expected to be a numpy array of raw text.
        vectorizer : Callable[..., Union[CountVectorizer, HashingVectorizer, TfidfVectorizer]]
            A scikit-learn vectorizer to be called/instatiated

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to the vectorizer during instantiation
        """

        self.vectorizer = vectorizer(**kwargs)

    def extract_features(
        self, X_train: Iterable[str], X_test: Iterable[str]
    ) -> Tuple[spmatrix, spmatrix]:
        """Extract the features from a text dataset.

        Parameters
        ----------
        X_train : np.ndarray
            A one dimensional array of textual training data.
        X_test : np.ndarray
            A one dimensional array of textual training data.

        Returns
        -------
        Tuple[spmatrix, spmatrix]
            Two dimensional sparse feature representations of the input corpora.
        """

        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        return X_train, X_test