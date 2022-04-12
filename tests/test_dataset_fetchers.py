"""Test the dataset fetchers.

TODO
----
-

FIXME
-----
-
"""

import types

import numpy as np
from scipy import sparse

from active_learning import dataset_fetchers
from active_learning.dataset_fetchers import disk
from active_learning.dataset_fetchers import huggingface


class TestInterface:
    def test_no_stream1(self):
        X_train, X_test, y_train, y_test, target_names = dataset_fetchers.get_dataset(
            "20NewsGroups", False, 0
        )

        assert isinstance(X_train, list)
        assert isinstance(X_test, list)
        assert sparse.isspmatrix_csr(y_train)
        assert sparse.isspmatrix_csr(y_test)
        assert isinstance(target_names, np.ndarray)

    def test_stream1(self):
        X_train, X_test, y_train, y_test, target_names = dataset_fetchers.get_dataset(
            "20NewsGroups", True, 0
        )

        assert isinstance(X_train, types.GeneratorType)
        assert isinstance(X_test, types.GeneratorType)
        assert sparse.isspmatrix_csr(y_train)
        assert sparse.isspmatrix_csr(y_test)
        assert isinstance(target_names, np.ndarray)

        assert isinstance(next(X_train), str)
        assert isinstance(next(X_test), str)


class TestTextFileDatasetFetcher:
    def test_fetch1(self):
        fetcher = disk.PredefinedTextFileDatasetFetcher(0, "20NewsGroups")
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

        assert isinstance(X_train, list)
        assert isinstance(X_test, list)
        assert sparse.isspmatrix_csr(y_train)
        assert sparse.isspmatrix_csr(y_test)
        assert isinstance(target_names, np.ndarray)

        assert len(X_train) == 9840
        assert y_train.shape[0] == 9840
        assert len(X_test) == 6871
        assert y_test.shape[0] == 6871

        assert isinstance(X_train[0], str)
        assert isinstance(X_test[0], str)

    def test_fetch2(self):
        fetcher = disk.RandomizedTextFileDatasetFetcher(
            0, "WebKB", categories=["course", "faculty", "project", "student"]
        )
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

        assert isinstance(X_train, list)
        assert isinstance(X_test, list)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert len(X_train) == 3147
        assert len(y_train) == 3147
        assert len(X_test) == 1049
        assert len(y_test) == 1049

        assert isinstance(X_train[0], str)
        assert isinstance(X_test[0], str)
        assert np.issubdtype(type(y_train[0]), np.integer)
        assert np.issubdtype(type(y_test[0]), np.integer)

    def test_fetch3(self):
        fetcher = disk.PredefinedTextFileDatasetFetcher(
            0,
            "Reuters",
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
        )
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

        assert isinstance(X_train, list)
        assert isinstance(X_test, list)
        assert sparse.isspmatrix_csr(y_train)
        assert sparse.isspmatrix_csr(y_test)
        assert isinstance(target_names, np.ndarray)

        assert len(X_train) == 6177
        assert y_train.shape[0] == 6177
        assert len(X_test) == 2406
        assert y_test.shape[0] == 2406

        assert isinstance(X_train[0], str)
        assert isinstance(X_test[0], str)

    def test_stream1(self):
        fetcher = disk.PredefinedTextFileDatasetFetcher(0, "20NewsGroups")
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()

        assert isinstance(X_train, types.GeneratorType)
        assert isinstance(X_test, types.GeneratorType)
        assert sparse.isspmatrix_csr(y_train)
        assert sparse.isspmatrix_csr(y_test)
        assert isinstance(target_names, np.ndarray)

        assert isinstance(next(X_train), str)
        assert isinstance(next(X_test), str)

    def test_stream2(self):
        fetcher = disk.RandomizedTextFileDatasetFetcher(
            0, "WebKB", categories=["course", "faculty", "project", "student"]
        )
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()

        assert isinstance(X_train, types.GeneratorType)
        assert isinstance(X_test, types.GeneratorType)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert isinstance(next(X_train), str)
        assert isinstance(next(X_test), str)

    def test_stream3(self):
        fetcher = disk.PredefinedTextFileDatasetFetcher(0, "20NewsGroups")
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()

        assert isinstance(X_train, types.GeneratorType)
        assert isinstance(X_test, types.GeneratorType)
        assert sparse.isspmatrix_csr(y_train)
        assert sparse.isspmatrix_csr(y_test)
        assert isinstance(target_names, np.ndarray)

        assert y_train.shape[0] == 9840
        assert y_test.shape[0] == 6871

        assert isinstance(next(X_train), str)
        assert isinstance(next(X_test), str)


class TestHuggingFace:
    def test_fetch1(self):
        fetcher = huggingface.PredefinedClassificationFetcher(0, "glue", name="sst2")
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

        assert isinstance(X_train, list)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, list)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert len(X_train) == 67349
        assert len(y_train) == 67349
        assert len(X_test) == 1821
        assert len(y_test) == 1821

        assert isinstance(X_train[0], str)
        assert isinstance(X_test[0], str)
        assert np.issubdtype(type(y_train[0]), np.integer)
        assert np.issubdtype(type(y_test[0]), np.integer)

    def test_stream1(self):
        fetcher = huggingface.PredefinedClassificationFetcher(0, "glue", name="sst2")
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()

        assert isinstance(X_train, types.GeneratorType)
        assert isinstance(X_test, types.GeneratorType)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert isinstance(next(X_train), str)
        assert isinstance(next(X_test), str)

    def test_randomized_test_size(self):
        fetcher = huggingface.RandomizedClassificationFetcher(0, "glue", name="sst2", test_size=42)
        X_train, X_test, y_train, y_test, _ = fetcher.fetch()

        assert len(X_train) == 67307
        assert len(y_train) == 67307
        assert len(X_test) == 42
        assert len(y_test) == 42
