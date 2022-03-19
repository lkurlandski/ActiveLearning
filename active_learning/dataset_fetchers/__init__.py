"""Get training and test data.

TODO
----
- RCV1, Reuters, WebKB are all multilabel classification tasks, which is currently not supported.
- Add multilabel support!

FIXME
-----
-
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Dict, Tuple

from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

from active_learning.dataset_fetchers.base import DatasetFetcher
from active_learning.dataset_fetchers.disk import (
    PredefinedPreprocessedFileDatasetFetcher,
    PredefinedTextFileDatasetFetcher,
)
from active_learning.dataset_fetchers.scikit_learn import ScikitLearnDatasetFetcher


# FIXME: the data in WebKB's raw directory is not formatted correctly
mapper: Dict[str, Tuple[DatasetFetcher, Dict[str, Any]]]
mapper = {
    "20NewsGroups": (PredefinedTextFileDatasetFetcher, {"dataset": "20NewsGroups"}),
    "Avila": (PredefinedPreprocessedFileDatasetFetcher, {"dataset": "Avila"}),
    "Covertype": (ScikitLearnDatasetFetcher, {"dataset": "Covertype"}),
    "Iris": (ScikitLearnDatasetFetcher, {"dataset": "Iris"}),
    # "RCV1_v2": (
    #     PredefinedTextFileDatasetFetcher,
    #     {
    #         "dataset": "RCV1_v2",
    #         "categories": [
    #             "CCAT",
    #             "GCAT",
    #             "MCAT",
    #             "C15",
    #             "ECAT",
    #             "M14",
    #             "C151",
    #             "C152",
    #             "GPOL",
    #             "M13",
    #         ],
    #     },
    # ),
    # "Reuters": (
    #     PredefinedTextFileDatasetFetcher,
    #     {
    #         "dataset": "Reuters",
    #         "categories": [
    #             "acq",
    #             "corn",
    #             "earn",
    #             "grain",
    #             "interest",
    #             "money-fx",
    #             "ship",
    #             "trade",
    #             "wheat",
    #         ],
    #     },
    # ),
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
