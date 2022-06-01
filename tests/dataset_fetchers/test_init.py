# pylint: disable=missing-module-docstring

from types import GeneratorType

from scipy import sparse

from active_learning.dataset_fetchers import get_dataset


def test_20newsgroups_multilabel_fetch():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "20newsgroups-multilabel", streaming=False, random_state=0
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


def test_20newsgroups_multilabel_stream():
    X_train, X_test, y_train, y_test, target_names = get_dataset(
        "20newsgroups-multilabel", streaming=True, random_state=0
    )

    assert isinstance(X_train, GeneratorType)
    assert isinstance(X_test, GeneratorType)
    assert sparse.isspmatrix_csr(y_train)
    assert sparse.isspmatrix_csr(y_test)
    assert isinstance(target_names, dict)

    assert len(list(X_train)) == 9840
    assert len(list(X_test)) == 6871
    assert y_train.shape[0] == 9840
    assert y_test.shape[0] == 6871
