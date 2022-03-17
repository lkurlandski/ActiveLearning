"""Get training and test data.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings

from datasets import load_dataset
import numpy as np
import sklearn.datasets
import sklearn.utils
from sklearn.model_selection import train_test_split

import stat_helper
import utils


class DatasetFetcher(ABC):
    """Retrieve the data required for learning."""

    def __init__(self, random_state: int) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state for reproducible results, when randomization needed
        """

        self.random_state = random_state

    @abstractmethod
    def fetch(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """Retrieve the data in memory.

        Returns
        -------
        Tuple[np.array, np.array, np.array, np.array, np.array]
            train data, test data, train labels, test labels, and target names
        """
        ...

    @abstractmethod
    def stream(self) -> Tuple[Any, Any, Any, Any, np.array]:
        """Retrieve the data using a memory-efficient streaming approach.

        Returns
        -------
        Tuple[Any, Any, Any, Any, np.array]
            train data, test data, train labels, test labels, and target names
        """
        ...


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


####################################################################################################


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


class ScikitLearnDatasetFetcher(DatasetFetcher):
    """Fetch datasets from scikit-learn."""

    def __init__(self, random_state: int, dataset: str, test_size: Union[float, int] = None):
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

        super().__init__(random_state)
        self.dataset = dataset
        self.test_size = test_size

    def fetch(self):

        loader, kwargs = self.get_dataset_loader_and_kwargs()

        if utils.check_callable_has_parameter(load_dataset, "random_state"):
            kwargs["random_state"] = self.random_state
        if utils.check_callable_has_parameter(load_dataset, "shuffle"):
            kwargs["shuffle"] = True

        if utils.check_callable_has_parameter(loader, "subset"):
            kwargs["subset"] = "train"
            bunch = loader(**kwargs)
            X_train, y_train = np.array(bunch["data"]), np.array(bunch["target"])
            target_names_train = np.array(bunch["target_names"])

            kwargs["subset"] = "test"
            bunch = loader(**kwargs)
            X_test, y_test = np.array(bunch["data"]), np.array(bunch["target"])
            target_names_test = np.array(bunch["target_names"])

            target_names = np.unique(np.concatenate((target_names_train, target_names_test)))
        else:
            bunch = loader(**kwargs)
            X, y = np.array(bunch["data"]), np.array(bunch["target"])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            target_names = np.array(bunch["target_names"])

        return X_train, X_test, y_train, y_test, target_names

    def stream(self):

        warnings.warn(
            "scikit-learn does not offer native streaming support. "
            "This request will not be serviced by streaming."
        )

        X_train, X_test, y_train, y_test, target_names = self.fetch()

        return X_train, X_test, y_train, y_test, target_names

    def get_dataset_loader_and_kwargs(
        self,
    ) -> Tuple[Callable[..., sklearn.utils.Bunch], Dict[str, Any]]:
        """Get the scikit-learn fetcher or loader function and its associated keyword arguments.

        Returns
        -------
        Tuple[Callable[..., sklearn.utils.Bunch], Dict[str, Any]]
            A scikit-learn fetcher or loader function and its associated keyword arguments.
        """

        if self.dataset == "Covertype":
            return sklearn.datasets.fetch_covtype, {}
        if self.dataset == "Iris":
            return sklearn.datasets.load_iris, {}
        if self.dataset == "20NewsGroups":
            return sklearn.datasets.fetch_20newsgroups, {"remove": ("headers", "footers", "quotes")}

        raise ValueError(f"{self.dataset} not recongized.")


# FIXME: the data in WebKB's raw directory is not formatted correctly
mapper: Dict[str, Tuple[DatasetFetcher, Dict[str, Any]]]
mapper = {
    "20NewsGroups": (PredefinedTextFileDatasetFetcher, {"dataset": "20NewsGroups"}),
    "Avila": (PredefinedPreprocessedFileDatasetFetcher, {"dataset": "Avila"}),
    "Covertype": (ScikitLearnDatasetFetcher, {"dataset": "Covertype"}),
    "Iris": (ScikitLearnDatasetFetcher, {"dataset": "Iris"}),
    "RCV1_v2": (
        PredefinedTextFileDatasetFetcher,
        {
            "dataset": "RCV1_v2",
            "categories": [
                "CCAT",
                "GCAT",
                "MCAT",
                "C15",
                "ECAT",
                "M14",
                "C151",
                "C152",
                "GPOL",
                "M13",
            ],
        },
    ),
    "Reuters": (
        PredefinedTextFileDatasetFetcher,
        {
            "dataset": "Reuters",
            "categories": [
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
        },
    ),
    # "WebKB": (
    #    RandomizedTextFileDatasetFetcher,
    #    {"dataset": "WebKB", "categories": ["student", "faculty", "course", "project"]},
    # ),
}


def get_dataset(
    dataset: str, stream: bool, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return any implemented dataset.

    Parameters
    ----------
    dataset : str
        Code to refer to a particular dataset
    random_state : int, optional
        Integer used for reproducible randomization
    stream : bool
        Controls whether data is streamed or returned in full

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories

    Raises
    ------
    ValueError
        If the dataset code is not recognized
    """

    if dataset not in mapper:
        raise ValueError(f"Dataset not recognized: {dataset}")

    fetcher_callable, kwargs = mapper[dataset]
    kwargs["random_state"] = random_state
    kwargs["dataset"] = dataset
    fetcher: DatasetFetcher = fetcher_callable(**kwargs)

    if stream:
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()
    else:
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

    return X_train, X_test, y_train, y_test, target_names


def test() -> None:
    """Test."""

    for dataset, (fetcher_callable, kwargs) in mapper.items():
        if dataset == "RCV1_v2":
            continue

        kwargs["random_state"] = 0
        print(dataset, "\n--------------------------------------------")
        fetcher = fetcher_callable(**kwargs)

        print("fetch()\n--------------------------------------------")
        pprint(
            [
                (type(r), r.shape, utils.format_bytes(utils.nbytes(r)), type(r[0]), str(r[0])[0:30])
                for r in fetcher.fetch()
            ],
            width=200,
        )

        print("stream()\n--------------------------------------------")
        pprint(
            [
                (type(r), r.shape, utils.format_bytes(utils.nbytes(r)), type(r[0]), str(r[0])[0:30])
                for r in fetcher.stream()
            ],
            width=200,
        )


if __name__ == "__main__":
    test()
