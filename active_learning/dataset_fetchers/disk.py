"""Get training and test data.

TODO
----
-

FIXME
-----
-
"""

from abc import abstractmethod
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import List, Tuple, Union
import warnings

import numpy as np
import sklearn.datasets
import sklearn.utils
from sklearn.model_selection import train_test_split

from active_learning import stat_helper
from active_learning.dataset_fetchers.base import DatasetFetcher


class FileDatasetFetcher(DatasetFetcher):
    """Retrieve data from files located on disk."""

    raw_path = Path("/projects/nlp-ml/io/input/raw/")

    def __init__(self, random_state: int, dataset: str) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        dataset : str
            Name of dataset to retrieve. Should be the same as code used throughout codebase and
                stored in raw_path
        """

        super().__init__(random_state)
        self.path = self.raw_path / dataset

    @abstractmethod
    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        ...

    @abstractmethod
    def stream(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        ...


class PreprocessedFileDatasetFetcher(FileDatasetFetcher):
    """Retrieve data from files located on disk when dataset is preprocessed and fully vectorized."""

    features = "X.csv"
    target = "y.csv"

    @abstractmethod
    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        ...

    def stream(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

        warnings.warn(
            "PreprocessedFileBasedFetcher does not offer native streaming support. "
            "This request will not be serviced by streaming."
        )

        X_train, X_test, y_train, y_test, target_names = self.fetch()

        return X_train, X_test, y_train, y_test, target_names


class PredefinedPreprocessedFileDatasetFetcher(PreprocessedFileDatasetFetcher):
    """Retrieve data from preprocessed datasets with a predefined train-test split."""

    def __init__(self, random_state: int, dataset: str) -> None:

        super().__init__(random_state, dataset)
        self.train_path = self.path / "train"
        self.test_path = self.path / "test"

    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

        X_train = np.loadtxt(self.train_path / self.features, delimiter=",")
        y_train = np.loadtxt(self.train_path / self.target, dtype=np.str_)
        X_test = np.loadtxt(self.test_path / self.features, delimiter=",")
        y_test = np.loadtxt(self.test_path / self.target, dtype=np.str_)

        X_train, y_train = stat_helper.shuffle_corresponding_arrays(
            X_train, y_train, self.random_state
        )

        target_names = np.unique(np.concatenate((y_train, y_test)))

        return X_train, X_test, y_train, y_test, target_names


class RandomizedPreprocessedFileDatasetFetcher(PreprocessedFileDatasetFetcher):
    """Retrieve data from preprocessed datasets that do not have a predefined train-test split."""

    def __init__(
        self, random_state: int, dataset: str, test_size: Union[float, int] = None
    ) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        dataset : str
            Name of dataset to retrieve
        test_size : Union[float, int], optional
            The size of the test set, which is either a floating point percentage of the total
                dataset size, or an integer number of examples to allocate to the test set,
                by default None, which uses 25% of the dataset as test data
        """

        super().__init__(random_state, dataset)
        self.test_size = test_size

    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

        X = np.loadtxt(self.path / self.features, delimiter=",")
        y = np.loadtxt(self.path / self.target, dtype=np.str_)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        target_names = np.unique(np.concatenate((y_train, y_test)))

        return X_train, X_test, y_train, y_test, target_names


class TextFileDatasetFetcher(FileDatasetFetcher):
    """Retrieve data from text files located on disk (classification tasks only)."""

    def __init__(self, random_state: int, dataset: str, categories: List[str] = None):
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        dataset : str
            Name of dataset to retrieve
        categories : List[str], optional
            Subset of ctegories to use, by default None, which indicates all categories used
        """

        super().__init__(random_state, dataset)
        self.categories = categories

    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

        X_train, X_test, y_train, y_test, target_names = self.load_files(load_content=True)
        return X_train, X_test, y_train, y_test, target_names

    def stream(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

        X_train, X_test, y_train, y_test, target_names = self.load_files(load_content=False)
        return X_train, X_test, y_train, y_test, target_names

    @abstractmethod
    def load_files(self, load_content: bool):
        """Load a classification dataset from files on disk.

        Parameters
        ----------
        load_content : bool
            Indicates if the content should be loaded into memory or not
        """
        ...


class PredefinedTextFileDatasetFetcher(TextFileDatasetFetcher):
    """Retrieve data from text files located on disk for datasets with predefined train/test splits."""

    def __init__(self, random_state: int, dataset: str, categories: List[str] = None):

        super().__init__(random_state, dataset, categories)

        self.train_path = self.path / "train"
        self.test_path = self.path / "test"

    def load_files(self, load_content: bool):

        key = "data" if load_content else "filenames"

        train_bunch = sklearn.datasets.load_files(
            self.train_path,
            categories=self.categories,
            load_content=load_content,
            random_state=self.random_state,
        )
        X_train, y_train = np.array(train_bunch[key]), np.array(train_bunch["target"])
        target_names_train = np.array(train_bunch["target_names"])

        test_bunch = sklearn.datasets.load_files(
            self.test_path,
            categories=self.categories,
            load_content=load_content,
            random_state=self.random_state,
        )
        X_test, y_test = np.array(test_bunch[key]), np.array(test_bunch["target"])
        target_names_test = np.array(test_bunch["target_names"])

        target_names = np.unique(np.concatenate((target_names_train, target_names_test)))

        return X_train, X_test, y_train, y_test, target_names


class RandomizedTextFileDatasetFetcher(TextFileDatasetFetcher):
    """Retrieve data from text files located on disk for datasets with no predefined splits."""

    def __init__(
        self,
        random_state: int,
        dataset: str,
        categories: List[str] = None,
        test_size: Union[float, int] = None,
    ):
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        dataset : str
            Name of dataset to retrieve
        categories : List[str], optional
            Subset of ctegories to use, by default None, which indicates all categories used
        test_size : Union[float, int], optional
            The size of the test set, which is either a floating point percentage of the total
                dataset size, or an integer number of examples to allocate to the test set,
                by default None, which uses 25% of the dataset as test data
        """

        super().__init__(random_state, dataset, categories)
        self.test_size = test_size

    def load_files(self, load_content: bool):

        key = "data" if load_content else "filenames"

        bunch = sklearn.datasets.load_files(
            self.path,
            categories=self.categories,
            load_content=load_content,
            random_state=self.random_state,
        )
        X, y = np.array(bunch[key]), np.array(bunch["target"])
        target_names = np.array(bunch["target_names"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.random_state, test_size=self.test_size
        )

        return X_train, X_test, y_train, y_test, target_names
