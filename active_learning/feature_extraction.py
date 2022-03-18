"""Extract features from complex data objects, such as text documents.

TODO
----
- Refactor the way streaming is passed to the extractors. The FeatureExtractor and 
    HuggingFaceFeatureExtractor do not need stream in their constructor. The 
    ScikitLearnFeatureExtractor does need stream in its constructor.
"""

from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from transformers import pipeline


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


class PreprocessedFeatureExtractor(FeatureExtractor):
    """Feature extractor for datasets which are already preprocessed and require no extraction."""

    def extract_features(self, X_train: Any, X_test: Any) -> Tuple[Any, Any]:
        """Return the input data object.

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

        return X_train, X_test


class ScikitLearnTextFeatureExtractor(FeatureExtractor):
    """Feature extractor using scikit-learn utilities."""

    def __init__(
        self,
        stream: bool,
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
        *kwargs : dict
            Keyword arguments passed to the vectorizer during instantiation

        Other Parameters
        ----------------
        input : str
            Used by the vectorizer, one of {'filename', 'file', 'content'}, defaults to 'content'
        encoding : str
            Used by the vectorizer, one of {'filename', 'file', 'content'}, defaults to 'utf-8'
        decode_error : str
            Used by the vectorizer, one of {'strict', 'ignore', 'replace'}, defaults to 'strict'
        """

        super().__init__(stream)

        kwargs["input"] = "filename" if self.stream else "content"
        kwargs["decode_error"] = "replace"
        self.vectorizer = vectorizer(**kwargs)

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

        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        return X_train, X_test


class HuggingFaceFeatureExtractor(FeatureExtractor):
    """Feature extractor using huggingface utilities."""

    def __init__(self, stream: bool, model: str) -> None:
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

        super().__init__(stream)

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
            text = doc
            if self.stream:
                with open(doc, "rb") as f:
                    doc = f.read()
                text = doc.decode("utf8", "replace")
            embeddings = self.vectorizer([text], padding=True, truncation=True)
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


mapper: Dict[str, Tuple[Callable[..., FeatureExtractor], Dict[str, Any]]]
mapper = {
    "preprocessed": (PreprocessedFeatureExtractor, {}),
    "count": (ScikitLearnTextFeatureExtractor, {"vectorizer": CountVectorizer}),
    "hash": (ScikitLearnTextFeatureExtractor, {"vectorizer": HashingVectorizer}),
    "tfidf": (ScikitLearnTextFeatureExtractor, {"vectorizer": TfidfVectorizer}),
    "bert": (HuggingFaceFeatureExtractor, {"model": "bert-base-uncased"}),
    "roberta": (HuggingFaceFeatureExtractor, {"model": "roberta-base"}),
}


def get_features(
    X_train: np.ndarray, X_test: np.ndarray, feature_representation: str, stream: bool
) -> np.ndarray:
    """Get numerical features from raw data; vectorize the data.

    Parameters
    ----------
    X_train : np.ndarray
        Raw training data to be vectorized
    X_train : np.ndarray
        Raw testing data to be vectorized
    feature_representation : str
        Code to refer to a feature representation
    stream : bool
        Controls whether incominng data is streamed or in full

    Returns
    -------
    np.ndarray
        Numeric representation of the data

    Raises
    ------
    ValueError
        If the feature representation is not recognized
    """

    if feature_representation not in mapper:
        raise ValueError(f"{feature_representation} not recognized.")

    vectorizer_callable, kwargs = mapper[feature_representation]
    vectorizer = vectorizer_callable(stream=stream, **kwargs)
    X_train, X_test = vectorizer.extract_features(X_train, X_test)

    return X_train, X_test
