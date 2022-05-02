"""Extract features from complex data objects, such as text documents.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
from scipy import sparse

from active_learning.feature_extractors.huggingface import HuggingFaceFeatureExtractor
from active_learning.feature_extractors.preprocessed import PreprocessedFeatureExtractor
from active_learning.feature_extractors.scikit_learn import ScikitLearnTextFeatureExtractor


valid_scikit_learn_reps = {
    "CountVectorizer",
    "HashingVectorizer",
    "TfidfVectorizer",
}


valid_huggingface_reps = {
    "albert-base-v2",
    "bert-base-uncased",
    "bert-cased",
    "distilbert-base-uncased",
    "distilroberta-base",
    "roberta-base",
}


valid_feature_reps = {"none"}.union(valid_scikit_learn_reps, valid_huggingface_reps)


def get_features(
    X_train: Iterable[Any], X_test: Iterable[Any], feature_rep: str
) -> Tuple[Union[np.ndarray, sparse.csr_matrix], Union[np.ndarray, sparse.csr_matrix]]:
    """Get numerical features from raw data; vectorize the data.

    Parameters
    ----------
    X_train : Iterable[Any]
        Raw training data to be vectorized.
    X_train : Iterable[Any]
        Raw testing data to be vectorized.
    feature_rep : str
        Code to refer to a feature representation.

    Returns
    -------
    Tuple[Union[np.ndarray, sparse.csr_matrix], Union[np.ndarray, sparse.csr_matrix]]
        Numeric representation of the train and test data.
    """
    if feature_rep == "none":
        feature_extractor = PreprocessedFeatureExtractor()
    elif feature_rep in valid_scikit_learn_reps:
        feature_extractor = ScikitLearnTextFeatureExtractor(feature_rep)
    elif feature_rep in valid_huggingface_reps:
        feature_extractor = HuggingFaceFeatureExtractor(feature_rep)
    else:
        raise ValueError(
            f"feature rep: {feature_rep} not recognized. "
            f"Valid representations are: {valid_feature_reps}"
        )

    X_train, X_test = feature_extractor.extract_features(X_train, X_test)

    return X_train, X_test
