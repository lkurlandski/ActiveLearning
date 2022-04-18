"""
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from active_learning.stopping_criteria.stabilizing_predictions import StabilizingPredictions


def test_binary():
    m = StabilizingPredictions(windows=3, threshold=0.67)

    m = m.update_from_preds(stop_set_preds=[1, 0, 0])
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

def test_multiclass():
    m = StabilizingPredictions(windows=3, threshold=0.67)

    m = m.update_from_preds(stop_set_preds=[1, 2, 3])
    assert not m.has_stopped
    assert np.isnan(m.agreement_scores[0])

    m = m.update_from_preds(stop_set_preds=[3, 1, 2])
    assert not m.has_stopped
    assert m.agreement_scores[-1] == -0.5

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


def test_multilabel():
    pass

def test_update_from_model_and_predict_sklearn():
    X, y = make_classification()
    clf = SVC()
    clf.fit(X, y)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
    )
    m = m.update_from_model_and_predict(
        model=clf,
        predict=lambda model, X: model.predict(X),
        stop_set=X[0:5]
    )

def test_update_tenorflow():
    pass

def test_update_pytorch():
    pass

def test_update_from_initial_unlabaled_pool_preds():
    X, y = make_classification()
    clf = SVC()
    clf.fit(X, y)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        stop_set_size=.5,
    )
    m = m.update_from_preds(initial_unlabeled_pool_preds=y)
