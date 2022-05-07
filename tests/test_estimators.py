"""Test the estimators module.
"""

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.multiclass import type_of_target

from active_learning import estimators


random_state = 0
n_samples = 103


def test_binary_classification():
    n_classes = 2
    X, y = datasets.make_classification(n_samples, n_classes=n_classes, random_state=random_state)
    clf = estimators.get_estimator("SVC", type_of_target(y), random_state)
    assert isinstance(clf, SVC)

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (n_samples, n_classes)


def test_multiclass_classification1():
    n_classes = 3
    X, y = datasets.make_classification(
        n_samples, n_classes=n_classes, n_informative=4, random_state=random_state
    )
    clf = estimators.get_estimator("SVC", type_of_target(y), random_state)
    assert not isinstance(clf, SVC)
    assert isinstance(clf, OneVsRestClassifier)

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (n_samples, n_classes)


def test_multiclass_classification2():
    n_classes = 5
    X, y = datasets.make_classification(
        n_samples, n_classes=n_classes, n_informative=5, random_state=random_state
    )
    clf = estimators.get_estimator("MLPClassifier", type_of_target(y), random_state)
    assert isinstance(clf, MLPClassifier)
    assert not isinstance(clf, OneVsRestClassifier)

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (n_samples, n_classes)


def test_multilabel_classification1():
    n_classes = 3
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=n_classes, sparse=True, random_state=random_state
    )
    clf = estimators.get_estimator("SVC", type_of_target(y), random_state)
    assert not isinstance(clf, SVC)
    assert isinstance(clf, OneVsRestClassifier)

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (n_samples, n_classes)


def test_multilabel_classification2():
    n_classes = 5
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=n_classes, sparse=True, random_state=random_state
    )
    clf = estimators.get_estimator("MLPClassifier", type_of_target(y), random_state)
    assert isinstance(clf, MLPClassifier)
    assert not isinstance(clf, OneVsRestClassifier)

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (n_samples, n_classes)


def test_multilabel_classification3():
    n_classes = 5
    X, y = datasets.make_multilabel_classification(
        n_samples, n_classes=n_classes, sparse=True, random_state=random_state
    )
    clf = estimators.get_estimator("RandomForestClassifier", type_of_target(y), random_state)
    assert isinstance(clf, RandomForestClassifier)
    MultiOutputToMultiLabelClassifier = estimators.get_multioutput_to_multilabel_wrapper(
        RandomForestClassifier
    )
    assert str(type(clf)) == str(MultiOutputToMultiLabelClassifier)
    assert not isinstance(clf, OneVsRestClassifier)

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    assert isinstance(probas, np.ndarray)
    assert probas.shape == (n_samples, n_classes)
