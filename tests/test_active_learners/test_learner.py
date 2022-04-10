"""Test the active learning procedure.

TODO
----
-

FIXME
-----
-
"""

from pathlib import Path
from typing import List

from scipy import sparse
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from modAL.batch import uncertainty_batch_sampling

from active_learning import estimators
from active_learning.active_learners import output_helper
from active_learning.active_learners import learn


random_state = 0
batch_size = 13
n_samples = 103
query_strategy = uncertainty_batch_sampling


def test_binary_classification(tmp_path):
    X, y = datasets.make_classification(n_samples, n_classes=2, random_state=random_state)
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)

def test_multiclass_classification(tmp_path):
    X, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)

def test_multiclass_classification_sparse_features(tmp_path):
    X, y = datasets.make_classification(
        n_samples, n_classes=3, n_informative=4, random_state=random_state
    )
    X = sparse.csr_matrix(X)
    estimator = DecisionTreeClassifier(random_state=random_state)
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.npy")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)

def test_multilabel_classification(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, sparse=True, random_state=random_state
    )
    estimator = SVC(kernel="linear", probability=True, random_state=random_state)
    estimator = OneVsRestClassifier(estimator)
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)

def test_multilabel_classification_sparse_features(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, sparse=True, random_state=random_state
    )
    MultiOutputToMultiLabelClassifier = estimators.get_MultiOutputToMultiLabelClassifier(
        RandomForestClassifier
    )
    estimator = MultiOutputToMultiLabelClassifier(random_state=random_state)
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)

def test_multilabel_classification_sparse_labels(tmp_path):
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=3, return_indicator="sparse", random_state=random_state
    )
    estimator = SVC(kernel="linear", probability=True, random_state=random_state)
    estimator = OneVsRestClassifier(estimator)
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.mtx")
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
    unlabeled_pool = learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.mtx")
    learn.learn(estimator, query_strategy, batch_size, unlabeled_pool)

def test_with_all_pools(tmp_path):
    pools: List[learn.Pool] = []
    for i in range(3):
        X, y = datasets.make_multilabel_classification(
            n_samples, n_classes=3, random_state=random_state + 1
        )
        path = Path(tmp_path) / str(i)
        path.mkdir(parents=False)
        pools.append(learn.Pool(X, y, tmp_path / "X.mtx", tmp_path / "y.mtx"))

    MultiOutputToMultiLabelClassifier = estimators.get_MultiOutputToMultiLabelClassifier(
        RandomForestClassifier
    )
    estimator = MultiOutputToMultiLabelClassifier(random_state=random_state)
    learn.learn(
        estimator,
        query_strategy,
        batch_size,
        unlabeled_pool=pools[0],
        test_set=pools[0],
        stop_set=pools[1],
    )

    assert pools[0].X_path.exists()
    assert pools[0].y_path.exists()
    assert pools[1].X_path.exists()
    assert pools[1].y_path.exists()
    assert pools[2].X_path.exists()
    assert pools[2].y_path.exists()
