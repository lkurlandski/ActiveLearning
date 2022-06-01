# pylint: disable=missing-module-docstring

from types import GeneratorType

import numpy as np
from scipy import sparse

from active_learning.dataset_fetchers.huggingface import get_dataset
from tests import random_state


def test_emotion_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "emotion", False, random_state=random_state
    )

    assert isinstance(X_train, list)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, list)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(X_train) == 16000
    assert len(X_test) == 2000
    assert len(y_train) == 16000
    assert len(y_test) == 2000
    assert len(target_names) == 6

    assert isinstance(X_train[0], str)
    assert isinstance(X_test[0], str)
    assert np.issubdtype(type(y_train[0]), np.integer)
    assert np.issubdtype(type(y_test[0]), np.integer)


def test_emotion_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "emotion", True, random_state=random_state
    )

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 16000
    assert len(list(X_test)) == 2000
    assert len(y_train) == 16000
    assert len(y_test) == 2000
    assert len(target_names) == 6


def test_glue_sst2_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "glue", False, random_state=random_state, name="sst2"
    )

    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(X_train) == 67349
    assert len(X_test) == 1821
    assert len(y_train) == 67349
    assert len(y_test) == 1821
    assert len(target_names) == 2

    assert isinstance(X_train[0], str)
    assert isinstance(X_test[0], str)
    assert np.issubdtype(type(y_train[0]), np.integer)
    assert np.issubdtype(type(y_test[0]), np.integer)


def test_glue_sst2_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "glue", True, random_state=random_state, name="sst2"
    )

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 67349
    assert len(list(X_test)) == 1821
    assert len(y_train) == 67349
    assert len(y_test) == 1821
    assert len(target_names) == 2


def test_glue_sst2_random_split_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "glue", False, random_state=random_state, test_size=42, name="sst2"
    )

    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(X_train) == 67349 - 42
    assert len(y_train) == 67349 - 42
    assert len(X_test) == 42
    assert len(y_test) == 42
    assert len(target_names) == 2


def test_glue_sst2_random_split_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "glue", True, random_state=random_state, test_size=42, name="sst2"
    )

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 67349 - 42
    assert len(list(X_test)) == 42
    assert len(y_train) == 67349 - 42
    assert len(y_test) == 42
    assert len(target_names) == 2
