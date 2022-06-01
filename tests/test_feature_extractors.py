"""
Test the feature extraction procedures.
"""

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups

from active_learning import feature_extractors
from active_learning.feature_extractors import (
    huggingface,
    preprocessed,
    scikit_learn,
    gensim_,
)


tr_corpus = fetch_20newsgroups(subset="train", return_X_y=True, shuffle=False)[0][0:100]

ts_corpus = fetch_20newsgroups(subset="test", return_X_y=True, shuffle=False)[0][0:75]


def tr_corpus_gen(n_samples=len(tr_corpus)):
    return (x for i, x in enumerate(tr_corpus) if i < n_samples)


def ts_corpus_gen(n_samples=len(ts_corpus)):
    return (x for i, x in enumerate(ts_corpus) if i < n_samples)


class TestInterface:
    def test_scikit_learn(self):
        tr, ts = feature_extractors.get_features(tr_corpus, ts_corpus, "CountVectorizer")
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)

    def test_huggingface(self):
        tr, ts = feature_extractors.get_features(tr_corpus[:4], ts_corpus[:4], "bert-base-uncased")
        assert isinstance(tr, np.ndarray)
        assert isinstance(ts, np.ndarray)

    def test_gensim(self):
        tr, ts = feature_extractors.get_features(tr_corpus, ts_corpus, "Word2Vec")
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
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor("CountVectorizer")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)

    def test2(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor("HashingVectorizer")
        tr, ts = vectorizer.extract_features(iter(tr_corpus), iter(ts_corpus))
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)

    def test3(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor("TfidfVectorizer")
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)


class TestScikitLearnTextFeatureExtractorStreaming:
    def test1(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor("CountVectorizer")
        tr, ts = vectorizer.extract_features(tr_corpus_gen(), ts_corpus_gen())
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)

    def test2(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor("HashingVectorizer")
        tr, ts = vectorizer.extract_features(tr_corpus_gen(), ts_corpus_gen())
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)

    def test3(self):
        vectorizer = scikit_learn.ScikitLearnTextFeatureExtractor("TfidfVectorizer")
        tr, ts = vectorizer.extract_features(iter(tr_corpus_gen()), iter(ts_corpus_gen()))
        assert sparse.isspmatrix_csr(tr)
        assert sparse.isspmatrix_csr(ts)


class TestHuggingFaceFeatureExtractor:
    n_samples = 4

    def test1(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor("bert-base-uncased")
        tr, ts = vectorizer.extract_features(
            tr_corpus[: self.n_samples], ts_corpus[: self.n_samples]
        )
        assert tr.shape == (self.n_samples, 768)
        assert ts.shape == (self.n_samples, 768)

    def test2(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor("roberta-base")
        tr, ts = vectorizer.extract_features(
            tr_corpus[: self.n_samples], ts_corpus[: self.n_samples]
        )
        assert tr.shape == (self.n_samples, 768)
        assert ts.shape == (self.n_samples, 768)


class TestHuggingFaceFeatureExtractorStreaming:
    n_samples = 4

    def test1(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor("bert-base-uncased")
        tr, ts = vectorizer.extract_features(
            tr_corpus_gen(self.n_samples), ts_corpus_gen(self.n_samples)
        )
        assert tr.shape == (self.n_samples, 768)
        assert ts.shape == (self.n_samples, 768)

    def test2(self):
        vectorizer = huggingface.HuggingFaceFeatureExtractor("roberta-base")
        tr, ts = vectorizer.extract_features(
            tr_corpus_gen(self.n_samples), ts_corpus_gen(self.n_samples)
        )
        assert tr.shape == (self.n_samples, 768)
        assert ts.shape == (self.n_samples, 768)


class TestGensimFeatureExtractor:
    def test1(self):
        vectorizer = gensim_.GensimFeatureExtractor("FastText", vector_size=50)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (len(tr_corpus), 50)
        assert ts.shape == (len(ts_corpus), 50)

    def test2(self):
        vectorizer = gensim_.GensimFeatureExtractor("Doc2Vec", vector_size=50)
        tr, ts = vectorizer.extract_features(tr_corpus, ts_corpus)
        assert tr.shape == (len(tr_corpus), 50)
        assert ts.shape == (len(ts_corpus), 50)


class TestGensimFeatureExtractorStreaming:
    def test1(self):
        vectorizer = gensim_.GensimFeatureExtractor("FastText", vector_size=50)
        tr, ts = vectorizer.extract_features(tr_corpus_gen(), ts_corpus_gen())
        assert tr.shape == (len(tr_corpus), 50)
        assert ts.shape == (len(ts_corpus), 50)

    def test2(self):
        vectorizer = gensim_.GensimFeatureExtractor("Doc2Vec", vector_size=50)
        tr, ts = vectorizer.extract_features(tr_corpus_gen(), ts_corpus_gen())
        assert tr.shape == (len(tr_corpus), 50)
        assert ts.shape == (len(ts_corpus), 50)
