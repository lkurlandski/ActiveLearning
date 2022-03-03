"""Extract features from complex data objects, such as text documents.
"""

from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, List, Union

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import make_pipeline, Pipeline
from transformers import pipeline


class FeatureExtractor(ABC):
    """Tool for extracting features from complex data objects."""

    @abstractmethod
    def extract_features(self, X: Any):
        """Extract the features from a collection of data objects.

        Parameters
        ----------
        X : Any
            A dataset of complex objects
        """


class PreprocessedFeatureExtractor(FeatureExtractor):
    """Feature extractor for datasets which are already preprocessed and require no extraction."""

    def extract_features(self, X: Any):
        """Return the input data object.

        Parameters
        ----------
        X : Any
            A dataset of preprocessed examples, ready for learning
        """

        return X


class ScikitLearnTextFeatureExtractor(FeatureExtractor):
    """Feature extractor using scikit-learn utilities."""

    def __init__(
        self,
        vectorizer: Union[
            CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer, Pipeline
        ],
    ):
        """Create instance with a custom vectorizer.

        Parameters
        ----------
        vectorizer : Union[
            CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer, Pipeline
        ]
            A scikit-learn Transformer or a Pipeline containing a chain of Transformers
        """

        self.vectorizer = vectorizer

    def extract_features(self, X: np.ndarray):
        """Extract the features from a text dataset.

        Parameters
        ----------
        X : np.ndarray
            A one dimensional array containing strings of text
        """

        return self.vectorizer.fit_transform(X)


class HuggingFaceFeatureExtractor(FeatureExtractor):
    """Feature extractor using huggingface utilities."""

    def __init__(self, model: str):
        """Create instance with a specific pretrained transformer.

        Parameters
        ----------
        model : str
            Name of a model from huggingface's collection of pretrained models
        """

        self.vectorizer = pipeline(task="feature-extraction", model=model)

    def extract_features(self, X: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Extract the features from a text dataset.

        Parameters
        ----------
        X : np.ndarray
            A one dimensional array containing strings of text
        """

        if isinstance(X, np.ndarray):
            X = X.tolist()
        embeddings = self.vectorizer(X, padding=True, truncation=True)
        features = self.mean_text_embeddings(embeddings)

        return features

    @staticmethod
    def mean_text_embeddings(embeddings: List[List[List[float]]]) -> np.ndarray:
        """Generate a mean embedding for a single document, suitable for representing as 1D array.

        Parameters
        ----------
        embeddings : List[List[List[float]]]
            A two dimensional embedding space representing a string or document of tokens

        Returns
        -------
        np.ndarray
           A one dimensional numeric array which is the mean of the individual token emebddings.
        """

        array = []
        for x in embeddings:
            embedding = np.array(x[0])
            mean_embedding = np.mean(embedding, axis=0)
            array.append(mean_embedding)
        return np.array(array)


def get_features(X, feature_representation):

    if feature_representation == "preprocessed":
        vectorizer = PreprocessedFeatureExtractor()
    elif feature_representation == "count":
        # HashingVectorizer is essentially a more optimized CountVectorizer
        vectorizer = ScikitLearnTextFeatureExtractor(HashingVectorizer())
    elif feature_representation == "tfidf":
        # HashingVectorizer + TfidfTransformer is essentially an optimized TfidfVectorizer
        vectorizer = ScikitLearnTextFeatureExtractor(
            make_pipeline(HashingVectorizer(), TfidfTransformer())
        )
    elif feature_representation == "bert":
        vectorizer = HuggingFaceFeatureExtractor("bert-base-uncased")
    elif feature_representation == "roberta":
        vectorizer = HuggingFaceFeatureExtractor("roberta-base")
    else:
        raise ValueError(f"Feature representation was not recongized: {feature_representation}")

    features = vectorizer.extract_features(X)

    return features


def test() -> None:
    """Test."""

    X = [
        "Hi there!",
        "How are you?",
        "I'm clinically insane, how are you?",
        "Oh...uh I'm fine. I guess I'll be going n--",
        "No.",
        "What?",
        "No. You will not be going. You will never be going.",
    ]

    for feature_representation in ("count", "tfidf", "bert", "roberta"):
        print(f"Extracting features with {feature_representation}")
        features = get_features(X, feature_representation)
        print(f"{type(features)}")
    print("Done.")


if __name__ == "__main__":
    test()
