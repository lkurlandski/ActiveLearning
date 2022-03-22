"""
Test the feature extraction procedures.

TODO
----
- Add tests for the FeatureExtractor, PreprocessedFeatureExtractor.
- If the HuggingFaceFeatureExtractor's stream method is refactored, the associated unit tests can
    be made more efficient by not re-constructing the HuggingFaceFeatureExtractor for each test.

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

from active_learning.feature_extractors import (
    FeatureExtractor,
    PreprocessedFeatureExtractor,
    ScikitLearnTextFeatureExtractor,
    HuggingFaceFeatureExtractor,
)


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


class TestScikitLearnTextFeatureExtractor:
    def test1(self):
        vectorizer = ScikitLearnTextFeatureExtractor(vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test2(self):
        vectorizer = ScikitLearnTextFeatureExtractor(vectorizer=HashingVectorizer)
        tr, ts = vectorizer.extract_features(iter(tr_corpus), iter(ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test3(self):
        vectorizer = ScikitLearnTextFeatureExtractor(vectorizer=TfidfVectorizer)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)


class TestScikitLearnTextFeatureExtractorStreaming:
    def test1(self):
        vectorizer = ScikitLearnTextFeatureExtractor(vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test2(self):
        vectorizer = ScikitLearnTextFeatureExtractor(vectorizer=HashingVectorizer)
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test3(self):
        vectorizer = ScikitLearnTextFeatureExtractor(vectorizer=TfidfVectorizer)
        tr, ts = vectorizer.extract_features(
            iter((x for x in tr_corpus)), iter((x for x in ts_corpus))
        )
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)


class TestHuggingFaceFeatureExtractor:
    def test1(self):
        vectorizer = HuggingFaceFeatureExtractor(model="bert-base-uncased")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test2(self):
        vectorizer = HuggingFaceFeatureExtractor(model="roberta-base")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)


class TestHuggingFaceFeatureExtractorStreaming:
    def test1(self):
        vectorizer = HuggingFaceFeatureExtractor(model="bert-base-uncased")
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test2(self):
        vectorizer = HuggingFaceFeatureExtractor(model="roberta-base")
        tr, ts = vectorizer.extract_features((x for x in tr_corpus), (x for x in ts_corpus))
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)
