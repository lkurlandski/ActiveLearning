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
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from active_learning.feature_extractors.base import FeatureExtractor


class GensimFeatureExtractor(FeatureExtractor):
    """Feature extractor using gensim utilities."""
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
        if self.model == "w2v":
            return self.get_w2v_vectors(X_train, X_test)
        elif self.model == "d2v":
            return self.get_d2v_vectors(X_train, X_test), 
        else:
            pass
            #raise some error

    def get_w2v_vectors(self, X_train, X_test):

        X_train = list(X_train)
        X_test = list(X_test)

        X_train = [sentence.split() for sentence in X_train]
        X_test = [sentence.split() for sentence in X_test]
        X_combined = X_train + X_test

        vectors = Word2Vec(sentences=X_combined, min_count=1, window=15, epochs=50)
        
        X_train = self._vectorize_split(X_train, vectors)
        X_test = self._vectorize_split(X_test, vectors)
        print(np.asarray(X_test))

        return csr_matrix(np.asarray(X_train)), csr_matrix(np.asarray(X_test))

    def get_d2v_vectors(self, X_train, X_test):
        X_train_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
        X_test_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_test)]

        X_train_model = Doc2Vec(X_train_documents, vector_size=5, window=2, min_count=1)
        X_test_model = Doc2Vec(X_test_documents, vector_size=5, window=2, min_count=1)

        X_test_vectors = []
        X_train_vectors = []

        for document in range(0, (len(X_train_documents) - 1)):
            X_train_vectors.append(X_train_model[document])

        for document in range(0, (len(X_test_documents) - 1)):
            X_test_vectors.append(X_test_model[document])
        
        print("returning vectors")
        print(X_test_vectors)
        return np.asarray(X_train_vectors), np.asarray(X_test_vectors)


    def _vectorize_split(self, split, vectors):
        #this needs to be reworked to use a mean
        vectorized_iter = []
        vectorized = []
        token_cutoff = 100
        
        for sentence in split:
            for token in sentence[:token_cutoff]:
                try:
                    vectorized_iter.append(vectors.wv[token])
                    #look at the function in sequencer, add zeros and flatten
                except Exception:
                    print("Could not find token", token)

            last_pieces = token_cutoff - len(vectorized_iter)
            for i in range(last_pieces):
                vectorized_iter.append(np.zeros(100,))
            vectorized.append(np.asarray(vectorized_iter).flatten())
            vectorized_iter.clear()

        print(type(vectorized))
        return vectorized