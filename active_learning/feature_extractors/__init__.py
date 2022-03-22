"""Extract features from complex data objects, such as text documents.

TODO
----
- Refactor the way streaming is passed to the extractors. The FeatureExtractor and
    HuggingFaceFeatureExtractor do not need stream in their constructor. The
    ScikitLearnFeatureExtractor does need stream in its constructor.

FIXME
-----
-
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from active_learning.feature_extractors.base import FeatureExtractor
from active_learning.feature_extractors.huggingface import HuggingFaceFeatureExtractor
from active_learning.feature_extractors.preprocessed import PreprocessedFeatureExtractor
from active_learning.feature_extractors.scikit_learn import ScikitLearnTextFeatureExtractor


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
    vectorizer = vectorizer_callable(**kwargs)
    X_train, X_test = vectorizer.extract_features(X_train, X_test)

    return X_train, X_test
