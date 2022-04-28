"""
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from active_learning.stopping_criteria.stabilizing_predictions import StabilizingPredictions


def test_update_from_preds_stop_set_binary():
    m = StabilizingPredictions(windows=3, threshold=0.67, stop_set_size=3)

    m = m.update_from_preds(stop_set_preds=[1, 0, 0])
    assert m.agreement_metric == "kappa"
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[0])

    m = m.update_from_preds(stop_set_preds=[1, 0, 1])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0.4, 0.001)

    m = m.update_from_preds(stop_set_preds=[0, 0, 1])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0.4, 0.001)

    m = m.update_from_preds(stop_set_preds=[0, 0, 0])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 0

    m = m.update_from_preds(stop_set_preds=[0, 0, 0])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(1, 0.001)

    m = m.update_from_preds(stop_set_preds=[0, 0, 0])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(1, 0.001)

    m = m.update_from_preds(stop_set_preds=[0, 0, 0])
    assert m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(1, 0.001)

def test_update_from_preds_stop_set_multiclass():
    m = StabilizingPredictions(windows=3, threshold=0.67, stop_set_size=3)

    m = m.update_from_preds(stop_set_preds=[1, 2, 3])
    assert m.agreement_metric == "kappa"
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[0])

    m = m.update_from_preds(stop_set_preds=[3, 1, 2])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(-0.5, 0.001)

    m = m.update_from_preds(stop_set_preds=[3, 2, 1])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 0

    m = m.update_from_preds(stop_set_preds=[3, 2, 1])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m = m.update_from_preds(stop_set_preds=[3, 2, 1])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m = m.update_from_preds(stop_set_preds=[3, 2, 1])
    assert m.has_stopped
    assert m.agreement_scores[-1] == 1


def test_update_from_preds_stop_set_multilabel():
    m = StabilizingPredictions(windows=3, threshold=0.63, stop_set_size=3)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    assert m.agreement_metric == "alpha"
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[0])

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(1.0, .001)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 1, 0]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0, .001)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 1, 1], [0, 1, 1]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0.2007, .001)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 1, 1]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0.4444, .001)
    
    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 1, 1], [0, 1, 1]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0.4444, .001)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 1, 1]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(0.4444, .001)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 1, 1]])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(1.0, .001)

    m = m.update_from_preds(stop_set_preds=[[0, 0, 1], [0, 0, 1], [0, 1, 1]])
    assert m.has_stopped
    assert m.agreement_scores[-1] == pytest.approx(1.0, .001)


def test_update_from_preds_initial_unlabeled_pool_binary():
    X, y = make_classification()

    clf = SVC()
    clf.fit(X[:50], y[:50])
    preds = clf.predict(X)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=.5
    )
    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .5
    assert m.stop_set_indices.shape[0] == 50
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[-1])
    
    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .5
    assert m.stop_set_indices.shape[0] == 50
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .5
    assert m.stop_set_indices.shape[0] == 50
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .5
    assert m.stop_set_indices.shape[0] == 50
    assert m.has_stopped
    assert m.agreement_scores[-1] == 1

def test_update_from_preds_initial_unlabeled_pool_multiclass():
    X, y = make_classification(n_classes=3, n_informative=4)

    clf = SVC()
    clf.fit(X[:40], y[:40])
    preds = clf.predict(X)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=50
    )
    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == 50
    assert m.stop_set_indices.shape[0] == 50
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[-1])
    
    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == 50
    assert m.stop_set_indices.shape[0] == 50
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == 50
    assert m.stop_set_indices.shape[0] == 50
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == 50
    assert m.stop_set_indices.shape[0] == 50
    assert m.has_stopped
    assert m.agreement_scores[-1] == 1


def test_update_from_preds_initial_unlabeled_pool_multilabel():
    X, y = make_multilabel_classification()
    clf = OneVsRestClassifier(SVC())
    clf.fit(X[:50], y[:50])
    preds = clf.predict(X)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=.3
    )
    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .3
    assert m.stop_set_indices.shape[0] == 30
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[-1])
    
    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .3
    assert m.stop_set_indices.shape[0] == 30
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .3
    assert m.stop_set_indices.shape[0] == 30
    assert not m.has_stopped
    assert m.agreement_scores[-1] == 1

    m.update_from_preds(initial_unlabeled_pool_preds=preds)
    assert m.stop_set_size == .3
    assert m.stop_set_indices.shape[0] == 30
    assert m.has_stopped
    assert m.agreement_scores[-1] == 1


def test_update_from_model_binary():
    X, y = make_classification()
    clf = SVC()
    clf.fit(X, y)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=.5
    )
    m = m.update_from_model(
        model=clf,
        predict=lambda model, X: model.predict(X),
        stop_set=X[0:5]
    )
    assert m.agreement_metric == "kappa"


def test_update_from_model_multiclass():
    X, y = make_classification(n_informative=4, n_classes=3)
    clf = SVC()
    clf.fit(X, y)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=.5
    )
    m = m.update_from_model(
        model=clf,
        predict=lambda model, X: model.predict(X),
        stop_set=X[0:5]
    )
    assert m.agreement_metric == "kappa"


def test_update_from_model_multilabel():
    X, y = make_multilabel_classification()
    clf = OneVsRestClassifier(SVC())
    clf.fit(X, y)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=.5
    )
    m = m.update_from_model(
        model=clf,
        predict=lambda model, X: model.predict(X),
        stop_set=X[0:5]
    )
    assert m.agreement_metric == "alpha"
