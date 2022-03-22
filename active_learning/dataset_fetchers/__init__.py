"""Get training and test data.

TODO
----
- Add multilabel support for RCV1, Reuters, and WebKB

FIXME
-----
-
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Tuple

import numpy as np

from active_learning.dataset_fetchers import base, disk, scikit_learn, huggingface
from active_learning import utils


def get_mapper(random_state: int = 0) -> Callable[..., base.DatasetFetcher]:
    """Get the mapper to assist with instantiating the correct DatasetFetcher.

    Parameters
    ----------
    random_state : int, optional
        Random state for reporducible results, by default 0.

    Returns
    -------
    Callable[..., base.DatasetFetcher]
        A lambda function to instantiate the DatasetFetcher instance.
    """

    mapper = {
        "20NewsGroups": utils.init(
            disk.PredefinedTextFileDatasetFetcher,
            random_state=random_state,
            dataset="20NewsGroups",
        ),
        "Avila": utils.init(
            disk.PredefinedPreprocessedFileDatasetFetcher,
            random_state=random_state,
            dataset="Avila",
        ),
        "Covertype": utils.init(
            scikit_learn.ScikitLearnDatasetFetcher,
            random_state=random_state,
            dataset="Covertype",
        ),
        "Iris": utils.init(
            scikit_learn.ScikitLearnDatasetFetcher,
            random_state=random_state,
            dataset="Iris",
        ),
        "RCV1_v2": utils.init(
            disk.PredefinedTextFileDatasetFetcher,
            random_state=random_state,
            dataset="RCV1_v2",
            categories=[
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
        ),
        "Reuters": utils.init(
            disk.PredefinedTextFileDatasetFetcher,
            random_state=random_state,
            dataset="Reuters",
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
        ),
        "WebKB": utils.init(
            disk.RandomizedTextFileDatasetFetcher,
            random_state=random_state,
            dataset="WebKB",
            categories=["student", "faculty", "course", "project"],
        ),
        "glue": utils.init(
            huggingface.PredefinedClassificationFetcher,
            random_state=random_state,
            path="glue",
            name="sst2",
        ),
        "ag_news": utils.init(
            huggingface.PredefinedClassificationFetcher,
            random_state=random_state,
            path="ag_news",
        ),
        "amazon_polarity": utils.init(
            huggingface.PredefinedClassificationFetcher,
            random_state=random_state,
            path="amazon_polarity",
        ),
        "emotion": utils.init(
            huggingface.PredefinedClassificationFetcher,
            random_state=random_state,
            path="emotion",
        ),
    }

    return mapper


def get_dataset(
    dataset: str, stream: bool, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return any implemented dataset.

    Parameters
    ----------
    dataset : str
        Code to refer to a particular dataset
    stream : bool
        Controls whether data is streamed or returned in full
    random_state : int, optional
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories

    Raises
    ------
    ValueError
        If the dataset code is not recognized
    """

    mapper = get_mapper(random_state)

    if dataset not in mapper:
        raise ValueError(f"Dataset not recognized: {dataset}")

    fetcher = mapper[dataset]()

    if stream:
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()
    else:
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

    return X_train, X_test, y_train, y_test, target_names
