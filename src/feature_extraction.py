"""Extract features from complex data objects, such as text documents.

# 20Newsgroups memory data, lowest --mem success:
    count       16G
    hash        16G
    tfidf       16G
    hash-tfidf  16G
    bert        16G?
    roberta     16G?
"""

from abc import ABC, abstractmethod
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, List, Tuple, Union

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.pipeline import make_pipeline, Pipeline
from transformers import pipeline

import utils


class FeatureExtractor(ABC):
    """Tool for extracting features from complex data objects."""

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

    def extract_features(self, X_train: np.ndarray, X_test: np.ndarray):
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

    def __init__(self, model: str, chunk_size: int = 256) -> None:
        """Create instance with a specific pretrained transformer.

        Parameters
        ----------
        model : str
            Name of a model from huggingface's collection of pretrained models
        chunk_size : int, optional
            Number of transformer embeddings to take the mean of at a time, by default 3. Larger
                chunks require more memory but are more computationally efficient.
        """

        self.chunk_size = chunk_size
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
        for doc_chunk in utils.grouper(corpus, self.chunk_size, None):
            nanless_doc_chunk = [j for j in doc_chunk if j is not None]
            embeddings = self.vectorizer(nanless_doc_chunk, padding=True, truncation=True)
            processed_doc_embeddings = self.mean_text_embeddings(embeddings)
            X.extend(processed_doc_embeddings)

        return np.array(X)

    @staticmethod
    def mean_text_embeddings(embeddings: List[List[List[float]]]) -> List[np.ndarray]:
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

class GensimFeatureExtractor(FeatureExtractor):
    def __init__(self, model: str):
        #consider using the model string with a pipline
        self.model= model

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
        X_test = [sentence.split() for sentence in X_test]
        X_train = [sentence.split() for sentence in X_train]
        X_combined = X_test + X_train

        #X_combined_numpy = np.asarray(X_combined)
        #print(X_combined_numpy.shape[0])
        #print(X_test)
    
        vectors = Word2Vec(sentences=X_combined, min_count=1, window=15, epochs=50)
        #print(vectors.wv['integrity'])

        X_test_vectors = []
        X_train_vectors = []
        X_test_vectors_iter = []
        X_train_vectors_iter = []
        #create a list of lists of W2V vectors using X_test and X_train
        for sentence in X_test:
            for token in sentence[:300]:
                try:
                    X_test_vectors_iter.append(vectors.wv[token])
                    #look at the function in sequencer, add zeros and flatten
                except Exception:
                    print("Could not find token", token)

            last_pieces = 300 - len(X_test_vectors_iter)
            for i in range(last_pieces):
                X_test_vectors_iter.append(np.zeros(100,))
            X_test_vectors.append(np.asarray(X_test_vectors_iter).flatten())
            X_test_vectors_iter.clear()

        for sentence in X_train:
            for token in sentence[:300]:
                try:
                    X_train_vectors_iter.append(vectors.wv[token])
                    #look at the function in sequencer, add zeros and flatten
                except Exception:
                    print("Could not find token", token)
                    pass

            last_pieces = 300 - len(X_train_vectors_iter)
            for i in range(last_pieces):
                X_train_vectors_iter.append(np.zeros(100,))
            
            X_train_vectors.append(np.asarray(X_train_vectors_iter).flatten())
            X_train_vectors_iter.clear()

        #print(np.asarray(X_test_vectors).shape)
        #print(np.asarray(X_train_vectors).shape)

        return np.asarray(X_test_vectors), np.asarray(X_train_vectors)

valid_feature_representations = {
    "preprocessed",
    "count",
    "hash",
    "tfidf",
    "hash-tfidf",
    "bert",
    "roberta",
    "w2v"
}


def get_features(
    X_train: np.ndarray, X_test: np.ndarray, feature_representation: str
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

    Returns
    -------
    np.ndarray
        Numeric representation of the data

    Raises
    ------
    ValueError
        If the feature representation is not recognized
    """

    if feature_representation == "preprocessed":
        vectorizer = PreprocessedFeatureExtractor()
    elif feature_representation == "count":
        vectorizer = ScikitLearnTextFeatureExtractor(CountVectorizer())
    elif feature_representation == "hash":
        vectorizer = ScikitLearnTextFeatureExtractor(HashingVectorizer())
    elif feature_representation == "tfidf":
        vectorizer = ScikitLearnTextFeatureExtractor(HashingVectorizer())
    elif feature_representation == "hash-tfidf":
        vectorizer = ScikitLearnTextFeatureExtractor(
            make_pipeline(HashingVectorizer(), TfidfTransformer())
        )
    elif feature_representation == "bert":
        vectorizer = HuggingFaceFeatureExtractor("bert-base-uncased")
    elif feature_representation == "roberta":
        vectorizer = HuggingFaceFeatureExtractor("roberta-base")
    elif feature_representation == "w2v":
        vectorizer = GensimFeatureExtractor("w2v")
    else:
        raise ValueError(
            f"Feature representation was not recongized: {feature_representation}."
            f"Valid feature representations include: {valid_feature_representations}"
        )

    X_train, X_test = vectorizer.extract_features(X_train, X_test)

    return X_train, X_test


def test() -> None:
    """Test."""

    X_train = [
        "Hi there!",
        "How are you?",
        "I'm clinically insane, how are you?",
        "Oh...uh I'm fine. I guess I'll be going n--",
        "No.",
        "What?",
        "No. You will not be going. You will never be going.",
    ]
    X_test = ["This is some random test data.", "This data is for test set.", "This is some data"]

    for feature_representation in valid_feature_representations:
        print(f"Extracting features with {feature_representation}")
        _X_train, _X_test = get_features(X_train, X_test, feature_representation)
        print(f"{type(_X_train)}, {type(_X_test)}")
    print("Done.")


if __name__ == "__main__":
    test()
