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
from typing import Iterable, List, Tuple

import numpy as np
from transformers import pipeline

from active_learning.feature_extractors.base import FeatureExtractor


class HuggingFaceFeatureExtractor(FeatureExtractor):
    """Feature extractor using huggingface utilities."""

    def __init__(self, model: str, **kwargs) -> None:
        """Create instance with a specific pretrained transformer.

        Parameters
        ----------
        model : str
            Name of a model from huggingface's collection of pretrained models

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments to pass to huggingface pipeline() function.
        """

        self.vectorizer = pipeline(task="feature-extraction", model=model, **kwargs)

    def extract_features(
        self, X_train: Iterable[str], X_test: Iterable[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the features from a text dataset.

        Parameters
        ----------
        X_train : Iterable[str]
            A one dimensional iterable of textual training data.
        X_test : Iterable[str]
            A one dimensional iterable of textual test data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two dimensional feature representations of the input corpora.
        """

        X_train = self._extract_features(X_train)
        X_test = self._extract_features(X_test)

        return X_train, X_test

    def _extract_features(self, corpus: Iterable[str]) -> np.ndarray:
        """Extract the features from a corpus of textual data.

        Parameters
        ----------
        corpus : Iterable[str]
            Corpus of text to extract features from

        Returns
        -------
        np.ndarray
            Two dimensional feature representation of the corpus
        """

        X = []
        for doc in corpus:
            embeddings = self.vectorizer([doc], padding=True, truncation=True)
            embedding = self._mean_embeddings(embeddings)
            X.extend(embedding)

        return np.array(X)

    @staticmethod
    def _mean_embeddings(embeddings: List[List[List[float]]]) -> List[np.ndarray]:
        """Generate a mean embedding for a single document, suitable for representing as 1D array.

        Parameters
        ----------
        embeddings : List[List[List[float]]]
            A two dimensional embedding space representing a string or document of tokens

        Returns
        -------
        List[np.ndarray]
           A one dimensional numeric array which is the mean of the individual token emebddings.
        """

        array = []
        for x in embeddings:
            embedding = np.array(x[0])
            mean_embedding = np.mean(embedding, axis=0)
            array.append(mean_embedding)

        return array
