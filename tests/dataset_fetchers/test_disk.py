# pylint: disable=missing-module-docstring

from types import GeneratorType

from scipy import sparse

from active_learning.dataset_fetchers.disk import get_dataset
from tests import random_state


def test_20newsgroups_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "20newsgroups-multilabel", False, random_state=random_state
    )

    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(X_train) == 9840
    assert len(X_test) == 6871
    assert y_train.shape[0] == 9840
    assert y_test.shape[0] == 6871

    assert isinstance(X_train[0], str)
    assert isinstance(X_test[0], str)


def test_20newsgroups_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset("20newsgroups-multilabel", True)

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 9840
    assert len(list(X_test)) == 6871
    assert y_train.shape[0] == 9840
    assert y_test.shape[0] == 6871


def test_reuters_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "reuters",
        False,
        categories=[
            "acq",
            "corn",
            "earn",
            "grain",
            "interest",
            "money-fx",
            "ship",
            "trade",
            "wheat",
        ],
        random_state=random_state,
    )

    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(X_train) == 6177
    assert len(X_test) == 2406
    assert y_train.shape[0] == 6177
    assert y_test.shape[0] == 2406

    assert isinstance(X_train[0], str)
    assert isinstance(X_test[0], str)


def test_reuters_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "reuters",
        True,
        categories=[
            "acq",
            "corn",
            "earn",
            "grain",
            "interest",
            "money-fx",
            "ship",
            "trade",
            "wheat",
        ],
        random_state=random_state,
    )

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 6177
    assert len(list(X_test)) == 2406
    assert y_train.shape[0] == 6177
    assert y_test.shape[0] == 2406


def test_web_kb_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "web_kb",
        False,
        categories=["course", "faculty", "project", "student"],
        test_size=0.25,
        random_state=random_state,
    )

    assert isinstance(X_train, list)
    assert isinstance(X_test, list)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(X_train) == 3145
    assert len(X_test) == 1049
    assert y_train.shape[0] == 3145
    assert y_test.shape[0] == 1049

    assert isinstance(X_train[0], str)
    assert isinstance(X_test[0], str)


def test_web_kb_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "web_kb",
        True,
        categories=["course", "faculty", "project", "student"],
        test_size=0.25,
        random_state=random_state,
    )

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 3145
    assert len(list(X_test)) == 1049
    assert y_train.shape[0] == 3145
    assert y_test.shape[0] == 1049
