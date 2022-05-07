"""Extract features from complex data objects, such as text documents.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Iterable, List, Tuple, Union

import numpy as np
from nltk.tokenize import wordpunct_tokenize
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from active_learning.feature_extractors.base import FeatureExtractor


class GensimFeatureExtractor(FeatureExtractor):
    """Feature extractor using gensim utilities."""

    def __init__(self, model: str):
        # consider using the model string with a pipline
        self.model = model

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
        if self.model == "w2v" or self.model == "fasttext":
            return self.get_w2v_ft_vectors(X_train, X_test)
        elif self.model == "d2v":
            return self.get_d2v_vectors(X_train, X_test)

    #can maybe change the function name
    def get_w2v_ft_vectors(self, X_train: List[str], X_test: List[str]):
        """Extract the features using Word2Vec or FastText representation

        Parameters
        ----------
        X_train : List[str]
            A one dimensional iterable of textual training data.
        X_test : List[str]
            A one dimensional iterable of textual test data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two dimensional feature representations of the input corpora
        """

        X_train_tokens = self._tokenize(X_train)
        X_test_tokens = self._tokenize(X_test)

        if self.model == "w2v":
            model = Word2Vec(
                sentences=X_train_tokens, vector_size=500, min_count=1, window=15, epochs=20
            )
        else:
            model = FastText(
                sentences=X_train_tokens, vector_size=500, min_count=1, window=15, epochs=20
            )

        X_train_vectors = self._vectorize(model, X_train_tokens)
        X_test_vectors = self._vectorize(model, X_test_tokens)

        return X_train_vectors, X_test_vectors

    def get_d2v_vectors(self, X_train: List[str], X_test: List[str]):
        """Extract features using Doc2Vec representation

        Parameters
        ----------
        X_train : List[str]
            A one dimensional iterable of textual training data.
        X_test : List[str]
            A one dimensional iterable of textual test data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two dimensional feature representations of the input corpora
        """

        X_train_tokens = self._tokenize(X_train)
        X_test_tokens = self._tokenize(X_test)

        X_train_tagged_documents = [
            TaggedDocument(doc, [i]) for i, doc in enumerate(X_train_tokens)
        ]

        d2v_model = Doc2Vec(
            X_train_tagged_documents, vector_size=500, window=2, min_count=1, epochs=20
        )

        X_test_vectors = []
        X_train_vectors = []

        for doc_id in range(len(X_train_tokens)):
            X_train_vectors.append(d2v_model.infer_vector(X_train_tokens[doc_id]))

        for doc_id in range(len(X_test_tokens)):
            X_test_vectors.append(d2v_model.infer_vector(X_test_tokens[doc_id]))

        return np.array(X_train_vectors), np.array(X_test_vectors)

    def _vectorize(self, model: Union[FastText, Word2Vec], sentences: List[List[str]]):
        """Vectorizes the tokenized dataset.

        Parameters
        ----------
        model : Union[FastText, Word2Vec]
            Gensim model used to extract vector representation of a token.
        sentences : List[List[str]]
            Two tokenized dimensional representation of dataset.

        Returns
        -------
        np.ndarray
            Two dimensional vectorized representation of dataset.
        """

        sentence_vectors = []
        mean_vectors = []

        for sentence in sentences:

            for token in sentence:
                if token in model.wv.key_to_index or isinstance(model, FastText):
                    sentence_vectors.append(model.wv[token])
                else:
                    sentence_vectors.append(np.zeros(500))
                    print("here")

            if not sentence_vectors:
                mean_vectors.append(np.zeros(500))
            else:
                mean_vectors.append(np.mean(np.array(sentence_vectors), axis=0))
            sentence_vectors.clear()

        return np.array(mean_vectors)

    def _tokenize(self, sentences: List[str]) -> List[List[str]]:
        """Tokenizes the documents in the dataset.

        Parameters
        ----------
        sentences : List[str]
            A one dimensional iterable of textual data.

        Returns
        -------
        List[List[str]]
            Two dimensional tokenized representation of dataset.
        """

        tokens = []

        for sentence in sentences:
            tokens.append(wordpunct_tokenize(sentence))

        return tokens
