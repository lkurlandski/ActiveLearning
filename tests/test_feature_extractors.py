"""
Test the feature extraction procedures.

TODO
----
-

FIXME
-----
-
"""

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from active_learning import feature_extractors
from active_learning.feature_extractors import huggingface, preprocessed, scikit_learn


tr_corpus = [
    "This is the first document.",
    "This is the second second document.",
    "And the third one.",
    "Is this the first document?",
]

ts_corpus = [
    "This is some random test data.",
    "This data is for the test set.",
    "***. ### ???.",
]


class TestInterface:
    def test_no_stream1(self):
        tr, ts = feature_extractors.get_features(tr_corpus, ts_corpus, "count")
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test_no_stream2(self):
        tr, ts = feature_extractors.get_features(tr_corpus, ts_corpus, "bert")
        assert isinstance(tr, np.ndarray)
        assert isinstance(ts, np.ndarray)

    def test_stream1(self):
        tr, ts = feature_extractors.get_features(
            (i for i in tr_corpus), (i for i in ts_corpus), "count"
        )
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test_stream2(self):
        tr, ts = feature_extractors.get_features(
            (i for i in tr_corpus), (i for i in ts_corpus), "bert"
        )
        assert isinstance(tr, np.ndarray)
        assert isinstance(ts, np.ndarray)


class TestPreprocessedFeatureExtractor:
    def test1(self):
        vectorizer = preprocessed.PreprocessedFeatureExtractor()
        tr, ts = vectorizer.extract_features((i for i in tr_corpus), (i for i in ts_corpus))
        assert isinstance(tr, np.ndarray)
        assert isinstance(ts, np.ndarray)

    def test2(self):
        vectorizer = preprocessed.PreprocessedFeatureExtractor()
        tr, ts = vectorizer.extract_features([i for i in tr_corpus], [i for i in ts_corpus])
        assert isinstance(tr, list)
        assert isinstance(ts, list)


class TestScikitLearnTextFeatureExtractor:
    def test1(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor(vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test2(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor(vectorizer=HashingVectorizer)
        tr, ts = vectorizer.extract_features(iter(tr_corpus), iter(ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test3(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor(vectorizer=TfidfVectorizer)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)


class TestScikitLearnTextFeatureExtractorStreaming:
    def test1(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor(vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test2(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor(vectorizer=HashingVectorizer)
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test3(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor(vectorizer=TfidfVectorizer)
        tr, ts = vectorizer.extract_features(
            iter((x for x in tr_corpus)), iter((x for x in ts_corpus))
        )
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)


class TestHuggingFaceFeatureExtractor:
    def test1(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor(model="bert-base-uncased")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test2(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor(model="roberta-base")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)


class TestHuggingFaceFeatureExtractorStreaming:
    def test1(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor(model="bert-base-uncased")
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test2(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor(model="roberta-base")
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)
