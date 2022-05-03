"""Extract features using scikit-learn vectorizers.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Iterable, Tuple

from scipy import sparse
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
        vectorizer: str,
        **kwargs,
    ) -> None:
        """Create instance with a custom vectorizer.

        Parameters
        ----------
        vectorizer : str
            A scikit-learn vectorizer to be called/instatiated. One of:
                {'CountVectorizer', 'HashingVectorizer', 'TfidfVectorizer'}

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to the vectorizer during instantiation
        """
        if vectorizer == "CountVectorizer":
            self.vectorizer = CountVectorizer(**kwargs)
        elif vectorizer == "HashingVectorizer":
            self.vectorizer = HashingVectorizer(**kwargs)
        elif vectorizer == "TfidfVectorizer":
            self.vectorizer = TfidfVectorizer(**kwargs)
        else:
            raise ValueError(f"Unknown vectorizer: {vectorizer}")

    def extract_features(
        self, X_train: Iterable[str], X_test: Iterable[str]
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Extract the features from a text dataset.

        Parameters
        ----------
        X_train : np.ndarray
            A one dimensional array of textual training data.
        X_test : np.ndarray
            A one dimensional array of textual training data.

        Returns
        -------
        Tuple[sparse.csr_matrix, sparse.csr_matrix]
            Two dimensional sparse feature representations of the input corpora.
        """

        X_train: sparse.csr_matrix = self.vectorizer.fit_transform(X_train)
        X_test: sparse.csr_matrix = self.vectorizer.transform(X_test)

        # Sorting the indices is very important for usage in downstream parallelization tasks
        # This can be read about in scikit-learn's issue #6614
        X_train.sort_indices()
        X_test.sort_indices()

        return X_train, X_test
