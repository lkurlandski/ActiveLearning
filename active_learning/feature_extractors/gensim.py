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
import collections

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
            return self.get_d2v_vectors(X_train, X_test)
        else:
            pass
            #raise some error

    def get_w2v_vectors(self, X_train, X_test):

        X_train_tokens = [sentence.split() for sentence in X_train]
        X_test_tokens = [sentence.split() for sentence in X_test]

        w2v_model = Word2Vec(sentences=X_train, min_count=1, window=15, epochs=20)

        X_train_vectors = []
        X_train_sentence_vectors = []
        for sentence in X_train_tokens:
            for token in sentence:
                try:
                    X_train_sentence_vectors.append(w2v_model.wv[token])
                except Exception:
                    pass
            X_train_vectors.append(np.mean(np.asarray(X_train_sentence_vectors), axis=0))
            
        print("This is the shape of X_test_vectors", X_train_vectors.shape)

        X_test_vectors = []
        X_test_sentence_vectors = []
        for sentence in X_test_tokens:
            for token in sentence:
                try:
                    X_test_vectors.append(w2v_model.wv[token])
                except Exception:
                    pass
            X_test_vectors.append(np.mean(np.asarray(X_test_sentence_vectors), axis=0))
        
        print("This is the shape of X_test_vectors", X_test_vectors.shape)
        return X_train_vectors, X_test_vectors

    def get_d2v_vectors(self, X_train, X_test):

        print("this is the type", type(X_train))
        X_train_tokens = [sentence.split() for sentence in X_train]
        X_test_tokens = [sentence.split() for sentence in X_test]
        
        X_train_tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train_tokens)]

        print("this is the type", type(X_train_tagged_documents))
        d2v_model = Doc2Vec(X_train_tagged_documents, vector_size=800, window=2, min_count=1, epochs=20)

        X_test_vectors = []
        X_train_vectors = []

        for doc_id in range(len(X_train_tokens)):
            X_train_vectors.append(d2v_model.infer_vector(X_train_tokens[doc_id]))
        
        for doc_id in range(len(X_test_tokens)):
            X_test_vectors.append(d2v_model.infer_vector(X_test_tokens[doc_id]))
        
        print(np.asarray(X_test_vectors).shape)
        print(np.asarray(X_train_vectors).shape)
        return np.asarray(X_train_vectors), np.asarray(X_test_vectors)