"""Fetch datasets using the huggingface API.

TODO
----
-

FIXME
-----
- For glue, sst2, y_test contains only -1, but y_train contains 0 and 1.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Generator, List, Tuple, Union
import warnings

from datasets import load_dataset
import numpy as np

from active_learning.dataset_fetchers.base import DatasetFetcher


class HuggingFaceDatasetFetcher(DatasetFetcher):
    """Fetch datasets from huggingface."""

    def __init__(self, random_state: int, path: str, **kwargs) -> None:
        """Instantiate the fetcher.

        Parameters
        ----------
        random_state : int
            Random state to use for reproducibility.
        path : str
            Name of the dataset, recognized by huggingface.

        Other Parameters
        ----------------
        name : str
            Many hugging face datasets actually are collections of datasets. The name parameter
                selects a particular sub-dataset from a dataset collection.

        Raises
        ------
        ValueError
            System is not designed for the user to load individual train or test splits
        """

        if "split" in kwargs:
            raise ValueError(f"Loading an individual split is not supported: {kwargs['split']}.")

        super().__init__(random_state)
        self.path = path
        self.dataset = load_dataset(path, **kwargs)


class ClassificationFetcher(HuggingFaceDatasetFetcher):
    """Fetch hugging face datasets in classification format."""

    # Contains the name of the feature to use as data
    feature_keys = {
        "glue": "sentence",
        "ag_news": "text",
        "amazon_polarity": "content",  # Also contains a "title" field, which may be useful
        "emotion": "text",
    }

    def fetch(self) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:

        X_train, X_test, y_train, y_test, target_names = self.stream()

        return (
            list(X_train),
            list(X_test),
            np.array(list(y_train)),
            np.array(list(y_test)),
            target_names,
        )

    def stream(
        self,
    ) -> Tuple[
        Generator[Any, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        Generator[Any, None, None],
        np.ndarray,
    ]:

        warnings.warn("Performing pseudo-streaming for the target vector.")

        X_train = (x[self.feature_keys[self.path]] for x in self.dataset["train"])
        X_test = (x[self.feature_keys[self.path]] for x in self.dataset["test"])

        y_train = [x["label"] for x in self.dataset["train"]]
        y_test = [x["label"] for x in self.dataset["test"]]
        target_names = np.sort(np.unique(np.concatenate((np.array(y_train), np.array(y_test)))))

        y_train = (y for y in y_train)
        y_test = (y for y in y_test)

        return X_train, X_test, y_train, y_test, target_names


class PredefinedClassificationFetcher(ClassificationFetcher):
    """Fetcher for datasets with a predefined train/test split."""

    ...


class RandomizedClassificationFetcher(ClassificationFetcher):
    """Fetcher for datasets with no predefined train/test splits."""

    def __init__(
        self, random_state: int, path: str, test_size: Union[int, float] = None, **kwargs
    ) -> None:
        super().__init__(random_state, path, **kwargs)
        self.dataset = self.dataset["train"].train_test_split(
            test_size=test_size, seed=self.random_state
        )
