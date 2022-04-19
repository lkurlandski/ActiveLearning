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
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from active_learning.feature_extractors import base
from active_learning.feature_extractors import huggingface
from active_learning.feature_extractors import preprocessed
from active_learning.feature_extractors import scikit_learn
from active_learning.feature_extractors import gensim
from active_learning import utils


mapper: Dict[str, Callable[..., base.FeatureExtractor]]
mapper = {
    "preprocessed": utils.init(preprocessed.PreprocessedFeatureExtractor),
    "count": utils.init(scikit_learn.ScikitLearnTextFeatureExtractor, vectorizer=CountVectorizer),
    "hash": utils.init(scikit_learn.ScikitLearnTextFeatureExtractor, vectorizer=HashingVectorizer),
    "tfidf": utils.init(scikit_learn.ScikitLearnTextFeatureExtractor, vectorizer=TfidfVectorizer),
    "bert": utils.init(huggingface.HuggingFaceFeatureExtractor, model="bert-base-uncased"),
    "roberta": utils.init(huggingface.HuggingFaceFeatureExtractor, model="roberta-base"),
    "w2v": utils.init(gensim.GensimFeatureExtractor, model="w2v"),
    "d2v": utils.init(gensim.GensimFeatureExtractor, model="d2v"),
    "distilbert": utils.init(
        huggingface.HuggingFaceFeatureExtractor, model="distilbert-base-uncased"
    ),
    "bert-cased": utils.init(huggingface.HuggingFaceFeatureExtractor, model="bert-base-cased"),
    "albert": utils.init(huggingface.HuggingFaceFeatureExtractor, model="albert-base-v2"),
    "distilroberta": utils.init(
        huggingface.HuggingFaceFeatureExtractor, model="distilroberta-base"
    ),
}


def get_features(
    X_train: Iterable[Any], X_test: Iterable[Any], feature_representation: str
) -> Tuple[Union[np.ndarray, sparse.csr_matrix], Union[np.ndarray, sparse.csr_matrix]]:
    """Get numerical features from raw data; vectorize the data.

    Parameters
    ----------
    X_train : Iterable[Any]
        Raw training data to be vectorized.
    X_train : Iterable[Any]
        Raw testing data to be vectorized.
    feature_representation : str
        Code to refer to a feature representation.

    Returns
    -------
    Tuple[Union[np.ndarray, sparse.csr_matrix], Union[np.ndarray, sparse.csr_matrix]]
        Numeric representation of the train and test data.

    Raises
    ------
    KeyError
        If the feature representation is not recognized.
    """

    if feature_representation not in mapper:
        raise KeyError(f"{feature_representation} not recognized.")

    feature_extractor = mapper[feature_representation]()
    X_train, X_test = feature_extractor.extract_features(X_train, X_test)

    return X_train, X_test
