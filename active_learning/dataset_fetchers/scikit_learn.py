"""Fetch datasets using the scikit-learn API.

TODO
----
- Add support for fetching datasets from OpenML.org, which can be done through scikit-learn.

FIXME
-----
-
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Tuple, Union
import warnings

import numpy as np
import sklearn.datasets
import sklearn.utils
from sklearn.model_selection import train_test_split

from active_learning import utils
from active_learning.dataset_fetchers.base import DatasetFetcher


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

        if utils.check_callable_has_parameter(loader, "random_state"):
            kwargs["random_state"] = self.random_state
        if utils.check_callable_has_parameter(loader, "shuffle"):
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