"""Get training and test data.

TODO
----
- Determine if data should be extracted from raw or normalized then change dataset_path accordingly.
- Add support for multilabel classification in the TextDatasetFetchers

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

    datasets_path = Path("/projects/nlp-ml/io/input/raw/")

    def __init__(self, random_state: int, dataset: str) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        dataset : str
            Name of dataset to retrieve. Should be the same as code used throughout codebase and
                stored in datasets_path
        """

        super().__init__(random_state)
        self.path = self.datasets_path / dataset

    @abstractmethod
    def fetch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def stream(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...


class PreprocessedFileDatasetFetcher(FileDatasetFetcher):
    """Retrieve data from files located on disk when dataset is preprocessed/fully vectorized."""

    features = "X.csv"
    target = "y.csv"

    @abstractmethod
    def fetch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...

    def stream(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

    def fetch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

    def fetch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

    def fetch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        X_train, X_test, y_train, y_test, target_names = self.load_dataset(load_content=True)

        return X_train, X_test, y_train, y_test, target_names

    def stream(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        X_train, X_test, y_train, y_test, target_names = self.load_dataset(load_content=False)

        return X_train, X_test, y_train, y_test, target_names

    @abstractmethod
    def load_dataset(
        self, load_content: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a classification dataset from files on disk.

        Parameters
        ----------
        load_content : bool
            Indicates if the content should be loaded into memory or not
        """
        ...

    def _load_files(
        self, path: Path, categories: Union[List[str], None], load_content: bool
    ) -> Tuple[List[str], List[int], List[str]]:
        """Handle the process of actually extracting the files from their structure.

        The path must contain data in the correct format, which can be recursively structured.

        Parameters
        ----------
        path : Path
            Location of files on disk
        categories : Union[List[str], None]
            Subset of categories to select. If None, uses all categories found
        load_content : bool
            Whether or not to bring the contents of the files into memory or use path names

        Returns
        -------
        Tuple[List[str], List[int], List[str]]
            Features, labels, and target names found in the supplied path
        """

        def contains_directories(_path: Path) -> bool:
            """Determine if a directory's subdirectories contain only files (no directories) or not.

            Parameters
            ----------
            _path : Path
                Directory to check for this format

            Returns
            -------
            bool
                True if the _path contains subdirectories that contain directories
            """

            for subdir in _path.iterdir():
                for p in subdir.iterdir():
                    if p.is_dir():
                        return True

            return False

        if not contains_directories(path):
            bunch = sklearn.datasets.load_files(
                path,
                categories=categories,
                load_content=load_content,
                random_state=self.random_state,
            )

            X = bunch["data"] if load_content else bunch["filenames"]
            y = bunch["target"]
            target_names = bunch["target_names"]

            return X, y, target_names

        X = []
        y = []
        target_names = []
        i = 0
        for p in sorted(path.iterdir()):
            if categories is not None and p.name not in categories:
                continue

            X_, _, _ = self._load_files(p, None, load_content)
            y_ = [i for _ in range(len(X_))]
            X.extend(X_)
            y.extend(y_)
            target_names.append(p.name)
            i += 1

        return X, y, target_names

    def load_files(
        self, path: Path, categories: Union[List[str], None], load_content: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Handle the process of actually extracting the files from their structure.

        The path must contain data in the correct format.

        Parameters
        ----------
        path : Path
            Location of files on disk
        categories : Union[List[str], None]
            Subset of categories to select. If None, uses all categories found
        load_content : bool
            Whether or not to bring the contents of the files into memory or use path names

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Features, labels, and target names found in the supplied path
        """

        X, y, target_names = self._load_files(path, categories, load_content)

        return np.array(X), np.array(y), np.array(target_names)


class PredefinedTextFileDatasetFetcher(TextFileDatasetFetcher):
    """Retrieve data from text files located on disk for datasets with train/test splits."""

    def __init__(self, random_state: int, dataset: str, categories: List[str] = None):

        super().__init__(random_state, dataset, categories)
        self.train_path = self.path / "train"
        self.test_path = self.path / "test"

    def load_dataset(self, load_content: bool):

        X_train, y_train, target_names_train = self.load_files(
            self.train_path, self.categories, load_content
        )
        X_test, y_test, target_names_test = self.load_files(
            self.test_path, self.categories, load_content
        )
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

    def load_dataset(self, load_content: bool):

        X, y, target_names = self.load_files(self.path, self.categories, load_content)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.random_state, test_size=self.test_size
        )

        return X_train, X_test, y_train, y_test, target_names
