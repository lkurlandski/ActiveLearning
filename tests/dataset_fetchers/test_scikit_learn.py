# pylint: disable=missing-module-docstring

from types import GeneratorType

import numpy as np

from active_learning.dataset_fetchers.scikit_learn import get_dataset
from tests import random_state


def test_20newsgroups_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "20newsgroups-singlelabel", False, random_state=random_state
    )
    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(X_train) == 11314
    assert len(X_test) == 7532
    assert len(y_train) == 11314
    assert len(y_test) == 7532
    assert len(target_names) == 20


def test_20newsgroups_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "20newsgroups-singlelabel", True, random_state=random_state
    )
    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    X_train, X_test = list(X_train), list(X_test)

    assert len(X_train) == 11314
    assert len(X_test) == 7532
    assert len(y_train) == 11314
    assert len(y_test) == 7532
    assert len(target_names) == 20


def test_iris_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "iris", False, random_state=random_state, test_size=10
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(X_train) == 140
    assert len(y_train) == 140
    assert len(X_test) == 10
    assert len(y_test) == 10


def test_iris_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "iris", True, random_state=random_state, test_size=10
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(X_train) == 140
    assert len(X_test) == 10
    assert len(y_train) == 140
    assert len(y_test) == 10
