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
from active_learning.feature_extractors.gensim_ import GensimFeatureExtractor


valid_gensim_reps = {
    "fasttext-wiki-news-subwords-300",
    "conceptnet-numberbatch-17-06-300",
    "word2vec-ruscorpora-300",
    "word2vec-google-news-300",
    "glove-wiki-gigaword-50",
    "glove-wiki-gigaword-100",
    "glove-wiki-gigaword-200",
    "glove-wiki-gigaword-300",
    "glove-twitter-25",
    "glove-twitter-50",
    "glove-twitter-100",
    "glove-twitter-200",
    "Word2Vec",
    "Doc2Vec",
    "FastText",
}


valid_huggingface_reps = {
    "albert-base-v2",
    "bert-base-uncased",
    "bert-base-cased",
    "distilbert-base-uncased",
    "distilroberta-base",
    "roberta-base",
}


valid_scikit_learn_reps = {
    "CountVectorizer",
    "HashingVectorizer",
    "TfidfVectorizer",
}


valid_feature_reps = {"none"}.union(
    valid_scikit_learn_reps, valid_huggingface_reps, valid_gensim_reps
)


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
    elif feature_rep in {"Word2Vec", "FastText", "Doc2Vec"}:
        feature_extractor = GensimFeatureExtractor(feature_rep, vector_size=768, min_count=1)
    elif feature_rep in valid_gensim_reps:
        feature_extractor = GensimFeatureExtractor(feature_rep)
    else:
        raise ValueError(
            f"feature rep: {feature_rep} not recognized. "
            f"Valid representations are: {valid_feature_reps}"
        )

    X_train, X_test = feature_extractor.extract_features(X_train, X_test)

    return X_train, X_test
