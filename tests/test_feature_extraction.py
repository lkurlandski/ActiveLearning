"""
Test the feature extraction procedures.

TODO
----
- Add tests for the FeatureExtractor, PreprocessedFeatureExtractor.
- If the HuggingFaceFeatureExtractor's stream method is refactored, the associated unit tests can
    be made more efficient by not re-constructing the HuggingFaceFeatureExtractor for each test.
"""

from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from active_learning.feature_extraction import (
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

tr_files = [
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduusersadams",
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduuserscxh",
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduusersgzhang",
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduusersmarco",
]

ts_files = [
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduusersagapito",
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduusersdamani",
    "/projects/nlp-ml/io/input/raw/WebKB/student/texas/httpwww.cs.utexas.eduusershaizhou",
]


class TestScikitLearnTextFeatureExtractor:
    def test1(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=False, vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test2(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=False, vectorizer=HashingVectorizer)
        tr, ts = vectorizer.extract_features(iter(tr_corpus), iter(ts_corpus))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test3(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=False, vectorizer=TfidfVectorizer)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)


class TestScikitLearnTextFeatureExtractorStreaming:
    def test1(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=True, vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features(tr_files, ts_files)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test2(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=True, vectorizer=HashingVectorizer)
        tr, ts = vectorizer.extract_features(tr_files, ts_files)
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test3(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=True, vectorizer=TfidfVectorizer)
        tr, ts = vectorizer.extract_features(iter(tr_files), iter(ts_files))
        assert scipy.sparse.issparse(tr)
        assert scipy.sparse.issparse(ts)

    def test_not_vectorizing_the_paths_themselves(self):
        vectorizer = ScikitLearnTextFeatureExtractor(stream=False, vectorizer=CountVectorizer)
        vectorizer_stream = ScikitLearnTextFeatureExtractor(stream=True, vectorizer=CountVectorizer)
        tr, ts = vectorizer.extract_features(tr_files, ts_files)
        tr_streamed, ts_streamed = vectorizer_stream.extract_features(tr_files, ts_files)
        assert tr.shape != tr_streamed.shape
        assert ts.shape != ts_streamed.shape


class TestHuggingFaceFeatureExtractor:
    def test1(self):
        vectorizer = HuggingFaceFeatureExtractor(stream=False, model="bert-base-uncased")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test2(self):
        vectorizer = HuggingFaceFeatureExtractor(stream=False, model="roberta-base")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)


class TestHuggingFaceFeatureExtractorStreaming:
    def test1(self):
        vectorizer = HuggingFaceFeatureExtractor(stream=True, model="bert-base-uncased")
        tr, ts = vectorizer.extract_features(tr_files, ts_files)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test2(self):
        vectorizer = HuggingFaceFeatureExtractor(stream=True, model="roberta-base")
        tr, ts = vectorizer.extract_features(tr_files, ts_files)
        assert tr.shape == (4, 768)
        assert ts.shape == (3, 768)

    def test_not_vectorizing_the_paths_themselves(self):
        vectorizer = HuggingFaceFeatureExtractor(stream=False, model="bert-base-uncased")
        vectorizer_stream = HuggingFaceFeatureExtractor(stream=True, model="bert-base-uncased")
        tr, ts = vectorizer.extract_features(tr_files, ts_files)
        tr_streamed, ts_streamed = vectorizer_stream.extract_features(tr_files, ts_files)
        assert not np.array_equal(tr, tr_streamed)
        assert not np.array_equal(ts, ts_streamed)
