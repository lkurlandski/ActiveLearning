"""Get training and test data.

TODO
----
- Determine if data should be extracted from raw or normalized then change dataset_path accordingly.
- Add support for multilabel classification in the TextDatasetFetchers.
- If a subset of categories is selected, documents that do not belong in any of those selected
    categories are completely ignored. Is this the desired behavior? If not, should be ammended.
- Add a feature to control the size of the test set for datasets with predefined train test splits,
    eg, RCV1 has an enormous test set which is needlessly large and we do not need to use all of it.

FIXME
-----
-
"""

from abc import abstractmethod
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Generator, List, Tuple, Union
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
            Random state for reproducible results, when randomization needed.
        dataset : str
            Name of dataset to retrieve. Should be the same as code used throughout codebase and
                stored in datasets_path.
        """

        super().__init__(random_state)
        self.path = self.datasets_path / dataset


class PreprocessedFileDatasetFetcher(FileDatasetFetcher):
    """Retrieve data from files located on disk when dataset is preprocessed/fully vectorized."""

    features = "X.csv"
    target = "y.csv"

    def stream(
        self,
    ) -> Tuple[
        Generator[Any, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
    ]:

        warnings.warn("Performing pseudo-streaming instead of true streaming.")

        X_train, X_test, y_train, y_test, target_names = self.fetch()

        return (
            (x for x in X_train),
            (x for x in X_test),
            (y for y in y_train),
            (y for y in y_test),
            target_names,
        )


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
            Random state for reproducible results, when randomization needed.
        dataset : str
            Name of dataset to retrieve.
        test_size : Union[float, int], optional
            The size of the test set, which is either a floating point percentage of the total
                dataset size, or an integer number of examples to allocate to the test set,
                by default None, which uses 25% of the dataset as test data.
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
            Random state for reproducible results, when randomization needed.
        dataset : str
            Name of dataset to retrieve.
        categories : List[str], optional
            Subset of ctegories to use, by default None, which indicates all categories used.
        """

        super().__init__(random_state, dataset)
        self.categories = categories

    def fetch(
        self,
    ) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:

        X_train, X_test, y_train, y_test, target_names = self.load_dataset()

        return (
            list(X_train),
            list(X_test),
            np.array(list(y_train)),
            np.array(list(y_test)),
            np.array(target_names),
        )

    def stream(
        self,
    ) -> Tuple[
        Generator[str, None, None],
        Generator[str, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
    ]:

        X_train, X_test, y_train, y_test, target_names = self.load_dataset()

        return X_train, X_test, y_train, y_test, np.array(target_names)

    def recursively_load_files(
        self, path: Path, categories: Union[List[str], None]
    ) -> Tuple[List[str], List[int], List[str]]:
        """Handle the process of actually extracting the files from their structure.

        The path must contain data in the correct format, which can be recursively structured. This
            function is essentially a recursive version of sklearn.datasets.load_files.

        Parameters
        ----------
        path : Path
            Location of files on disk.
        categories : Union[List[str], None]
            Subset of categories to select. If None, uses all categories found.

        Returns
        -------
        Tuple[List[str], List[int], List[str]]
            Files of document, labels of the documents, and target names found in the supplied path
        """

        def contains_directories(path_: Path) -> bool:
            """Determine if a directory's subdirectories contain only files (no directories) or not.

            Parameters
            ----------
            path_ : Path
                Directory to check for this format

            Returns
            -------
            bool
                True if the path_ contains subdirectories that contain directories.
            """

            for subdir in path_.iterdir():
                for p in subdir.iterdir():
                    if p.is_dir():
                        return True

            return False

        if not contains_directories(path):
            bunch = sklearn.datasets.load_files(
                path,
                categories=categories,
                load_content=False,
                random_state=self.random_state,
            )

            return bunch["filenames"], bunch["target"], bunch["target_names"]

        X = []
        y = []
        target_names = []
        i = 0
        for p in sorted(path.iterdir()):
            if categories is not None and p.name not in categories:
                continue

            X_, _, _ = self.recursively_load_files(p, None)
            y_ = [i for _ in range(len(X_))]
            X.extend(X_)
            y.extend(y_)
            target_names.append(p.name)
            i += 1

        return X, y, target_names

    @staticmethod
    def X_y_to_generators(
        X: List[str], y: List[Any]
    ) -> Tuple[Generator[str, None, None], Generator[Any, None, None]]:
        """Convert a training corpus of filenames and its corresponding format to documents.

        Parameters
        ----------
        X : List[str]
            Filenames to read and return as generator.
        y : List[Any]
            Labels for the filenames.

        Returns
        -------
        Tuple[Generator[str, None, None], Generator[Any, None, None]]
            A generator of documents and a generator of labels.
        """

        return (open(f, "rb").read().decode("utf8", "replace") for f in X), (i for i in y)

    @abstractmethod
    def load_dataset(
        self,
    ) -> Tuple[
        Generator[Any, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
    ]:
        """Load a classification dataset from files on disk.

        Parameters
        ----------
        load_content : bool
            Indicates if the content should be loaded into memory or not.

        Returns
        -------
        Tuple[
                Generator[Any, None, None],
                Generator[Any, None, None],
                Generator[Any, None, None],
                Generator[Any, None, None],
                np.ndarray,
            ]
            Train data, test data, train labels, test labels, and target names.
        """
        ...


class PredefinedTextFileDatasetFetcher(TextFileDatasetFetcher):
    """Retrieve data from text files located on disk for datasets with train/test splits."""

    def __init__(self, random_state: int, dataset: str, categories: List[str] = None) -> None:

        super().__init__(random_state, dataset, categories)
        self.train_path = self.path / "train"
        self.test_path = self.path / "test"

    def load_dataset(
        self,
    ) -> Tuple[
        Generator[str, None, None],
        Generator[str, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
    ]:

        X_train, y_train, target_names_train = self.recursively_load_files(
            self.train_path, self.categories
        )
        X_test, y_test, target_names_test = self.recursively_load_files(
            self.test_path, self.categories
        )
        target_names = np.unique(np.concatenate((target_names_train, target_names_test)))

        X_train, y_train = self.X_y_to_generators(X_train, y_train)
        X_test, y_test = self.X_y_to_generators(X_test, y_test)

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

    def load_dataset(
        self,
    ) -> Tuple[
        Generator[str, None, None],
        Generator[str, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
    ]:

        X, y, target_names = self.recursively_load_files(self.path, self.categories)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.random_state, test_size=self.test_size
        )

        X_train, y_train = self.X_y_to_generators(X_train, y_train)
        X_test, y_test = self.X_y_to_generators(X_test, y_test)

        return X_train, X_test, y_train, y_test, target_names
