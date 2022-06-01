"""Test the active learning procedure.
"""

import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from modAL.batch import uncertainty_batch_sampling

from active_learning import estimators
from active_learning.al_pipeline import learn
from active_learning.al_pipeline.helpers import Pool


random_state = 0
batch_size = 13
n_samples = 103
query_strategy = uncertainty_batch_sampling


def test_binary_classification(tmp_path):
    X, y = datasets.make_classification(n_samples, n_classes=2, random_state=random_state)
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_multiclass_classification(tmp_path):
    X, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_multiclass_classification_sparse_features(tmp_path):
    X, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    X = sparse.csr_matrix(X)
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_multilabel_classification(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, sparse=True, random_state=random_state
    )
    estimator = SVC(kernel="linear", probability=True, random_state=random_state)
    estimator = OneVsRestClassifier(estimator)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_multilabel_classification_sparse_features(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, sparse=True, random_state=random_state
    )
    estimator = estimators.MultiOutputToMultiLabelClassifier(
        RandomForestClassifier(random_state=random_state)
    )
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_multilabel_classification_sparse_labels(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, return_indicator="sparse", random_state=random_state
    )
    estimator = SVC(kernel="linear", probability=True, random_state=random_state)
    estimator = OneVsRestClassifier(estimator)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_multilabel_classification_sparse_features_and_labels(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples,
        n_classes=3,
        sparse=True,
        return_indicator="sparse",
        random_state=random_state,
    )
    estimator = SVC(kernel="linear", probability=True, random_state=random_state)
    estimator = OneVsRestClassifier(estimator)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)


def test_with_all_pools(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, random_state=random_state + 1
    )
    path = tmp_path / "unlabeled_pool"
    path.mkdir()
    unlabeled_pool = Pool(X, y, X_path=path / "X.mtx", y_path=path / "y.mtx")

    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, random_state=random_state + 1
    )
    path = tmp_path / "test_set_pool"
    path.mkdir()
    test_set = Pool(X, y, X_path=path / "X.mtx", y_path=path / "y.mtx")

    estimator = estimators.MultiOutputToMultiLabelClassifier(
        RandomForestClassifier(random_state=random_state)
    )
    learn.learn(
        estimator,
        query_strategy,
        batch_size,
        unlabeled_pool,
        test_set=test_set,
    )

    assert unlabeled_pool.X_path.exists()
    assert unlabeled_pool.y_path.exists()
    assert test_set.X_path.exists()
    assert test_set.y_path.exists()


def test_get_first_batch_random():
    protocol = "random"

    _, y = datasets.make_classification(
        n_samples, n_classes=2, n_informative=4, random_state=random_state
    )
    idx = learn.get_first_batch(y, protocol, r=10)
    assert len(idx) == 10

    _, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    idx = learn.get_first_batch(y, protocol, r=10)
    assert len(idx) == 10

    _, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, random_state=random_state
    )
    idx = learn.get_first_batch(y, protocol, r=10)
    assert len(idx) == 10

    _, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, random_state=random_state, sparse=True
    )
    idx = learn.get_first_batch(y, protocol, r=10)
    assert len(idx) == 10


def test_get_first_batch_k_per_class():
    protocol = "k_per_class"

    _, y = datasets.make_classification(
        n_samples, n_classes=2, n_informative=4, random_state=random_state
    )
    idx = learn.get_first_batch(y, protocol, k=2)
    assert len(idx) == 2 * 2
    assert len(np.unique(y[idx])) == 2

    _, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    idx = learn.get_first_batch(y, protocol, k=2)
    assert len(idx) == 3 * 2
    assert len(np.unique(y[idx])) == 3

    _, y = datasets.make_multilabel_classification(
        n_samples, n_classes=4, random_state=random_state
    )
    idx = learn.get_first_batch(y, protocol, k=2)
    assert len(idx) == 4 * 2
    # TODO: add test to check that all classes are present in multilabel case

    _, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, random_state=random_state, sparse=True
    )
    idx = learn.get_first_batch(y, protocol, k=2)
    assert len(idx) == 3 * 2
    # TODO: add test to check that all classes are present in multilabel case


def test_early_stop_modes_exponential(tmp_path):

    X, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.npy")
    learn.learn(
        estimator, query_strategy, batch_size, unlabeled_pool, early_stop_mode="exponential"
    )


def test_early_stop_modes_finish(tmp_path):

    X, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = Pool(X, y, X_path=tmp_path / "X.mtx", y_path=tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool, early_stop_mode="finish")
