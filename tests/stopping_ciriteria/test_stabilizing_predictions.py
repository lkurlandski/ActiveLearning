"""
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from active_learning.stopping_criteria.stabilizing_predictions import StabilizingPredictions


def test_binary():
    m = StabilizingPredictions(windows=3, threshold=0.67)

    m = m.update(preds=[1, 0, 0])
    assert not m.is_stop()
    assert np.isnan(m.agreement_scores[0])

    m = m.update(preds=[1, 0, 1])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == pytest.approx(0.4, 0.001)

    m = m.update(preds=[0, 0, 1])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == pytest.approx(0.4, 0.001)

    m = m.update(preds=[0, 0, 0])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == 0

    m = m.update(preds=[0, 0, 0])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == pytest.approx(1, 0.001)

    m = m.update(preds=[0, 0, 0])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == pytest.approx(1, 0.001)

    m = m.update(preds=[0, 0, 0])
    assert m.is_stop()
    assert m.agreement_scores[-1] == pytest.approx(1, 0.001)

def test_multiclass():
    m = StabilizingPredictions(windows=3, threshold=0.67)

    m = m.update(preds=[1, 2, 3])
    assert not m.is_stop()
    assert np.isnan(m.agreement_scores[0])

    m = m.update(preds=[3, 1, 2])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == -0.5

    m = m.update(preds=[3, 2, 1])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == 0

    m = m.update(preds=[3, 2, 1])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == 1

    m = m.update(preds=[3, 2, 1])
    assert not m.is_stop()
    assert m.agreement_scores[-1] == 1

    m = m.update(preds=[3, 2, 1])
    assert m.is_stop()
    assert m.agreement_scores[-1] == 1


def test_multilabel():
    pass

def test_update_scikit_learn():
    X, y = make_classification()
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    m = StabilizingPredictions(
        windows=3,
        threshold=0.67,
        initial_unlabeled_pool=X,
        stop_set_size=100
    )
    m = m.update(model=clf, predict=lambda model, X: model.predict(X))

def test_update_tenorflow():
    pass

def test_update_pytorch():
    pass
