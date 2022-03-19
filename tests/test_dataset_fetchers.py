"""Test the dataset fetchers.

TODO
----
- Implement fixtures to test these classes (using real-world datasets will take long to execute).

FIXME
-----
-
"""


import numpy as np

from active_learning.dataset_fetchers.disk import (
    PredefinedTextFileDatasetFetcher,
    RandomizedTextFileDatasetFetcher,
)


class TestTextFileDatasetFetcher:
    def test_load_files1(self):
        fetcher = PredefinedTextFileDatasetFetcher(0, "20NewsGroups")
        categories = None
        X, y, target_names = fetcher.load_files(fetcher.test_path, categories, False)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert X.shape == (7532,)
        assert y.shape == (7532,)
        assert target_names.shape == (20,)

    def test_load_files2(self):
        fetcher = PredefinedTextFileDatasetFetcher(0, "20NewsGroups")
        categories = ["alt.atheism", "comp.os.ms-windows.misc", "talk.religion.misc"]
        X, y, target_names = fetcher.load_files(fetcher.test_path, categories, False)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert X.shape == (319 + 394 + 251,)
        assert y.shape == (319 + 394 + 251,)
        assert target_names.shape == (3,)

    def test_load_files3(self):
        fetcher = RandomizedTextFileDatasetFetcher(0, "WebKB")
        categories = ["course", "faculty", "project", "student"]
        X, y, target_names = fetcher.load_files(fetcher.path, categories, False)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(target_names, np.ndarray)

        assert X.shape == (928 + 1124 + 504 + 1640,)
        assert y.shape == (928 + 1124 + 504 + 1640,)
        assert target_names.shape == (4,)
