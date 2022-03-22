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
from typing import List, Tuple

import numpy as np
from transformers import pipeline

from active_learning.feature_extractors.base import FeatureExtractor


class HuggingFaceFeatureExtractor(FeatureExtractor):
    """Feature extractor using huggingface utilities."""

    def __init__(self, model: str) -> None:
        """Create instance with a specific pretrained transformer.

        Parameters
        ----------
        stream : bool
            Whether or not the input data will be streamed. In the case of text documents,
                streamed data is expected to be a numpy array of filenames containing raw text. Non-
                streamed data is expected to be a numpy array of raw text.
        model : str
            Name of a model from huggingface's collection of pretrained models
        """

        self.vectorizer = pipeline(task="feature-extraction", model=model)

    def extract_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the features from a text dataset.

        Parameters
        ----------
        X_train : np.ndarray
            A one dimensional array of textual training data
        X_test : np.ndarray
            A one dimensional array of textual training data

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two dimensional feature representations of the input corpora
        """

        X_train = self._extract_features(X_train)
        X_test = self._extract_features(X_test)

        return X_train, X_test

    def _extract_features(self, corpus: np.ndarray) -> np.ndarray:
        """Extract the features from a corpus of textual data.

        Parameters
        ----------
        corpus : np.ndarray
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
