"""Get training and test data.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Iterable, Tuple

import numpy as np

from active_learning.dataset_fetchers import base, disk, scikit_learn, huggingface
from active_learning import utils


valid_disk_datasets = {
    "20NewsGroups-multilabel",
    "Avila",
    "RCV1_v2",
    "Reuters",
    "WebKB",
}


valid_huggingface_datasets = {
    "ag_news",
    "amazon_polarity",
    "emotion",
    "glue",
    "imdb",
    "rotten_tomatoes",
    "tweet_eval",
}


valid_scikit_learn_datasets = {
    "20NewsGroups-singlelabel",
    "Covertype",
    "Iris",
}


valid_datasets = set().union(
    valid_disk_datasets,
    valid_huggingface_datasets,
    valid_scikit_learn_datasets
)


def get_dataset(
    dataset: str, stream: bool, random_state: int
) -> Tuple[Iterable[Any], Iterable[Any], np.ndarray, np.ndarray, np.ndarray]:
    """Return any implemented dataset.

    Parameters
    ----------
    dataset : str
        Code to refer to a particular dataset.
    stream : bool
        Controls whether data is streamed or returned in full.
    random_state : int, optional
        Integer used for reproducible randomization.

    Returns
    -------
    Tuple[Iterable[Any], Iterable[Any], np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, X_test, y_train, and y_test, along with the set of categories.

    Raises
    ------
    KeyError
        If the dataset code is not recognized.
    """
    # Datasets extracted from disk
    if dataset == "20NewsGroups-multilabel":
        fetcher = disk.PredefinedTextFileDatasetFetcher(
            random_state=random_state, dataset="20NewsGroups"
        )
    if dataset == "Avila":
        fetcher = disk.PredefinedPreprocessedFileDatasetFetcher(
            random_state=random_state,
            dataset="Avila",
        )
    if dataset == "RCV1_v2":
        fetcher = disk.RandomizedTextFileDatasetFetcher(
            random_state=random_state,
            dataset="RCV1_v2/train",
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
        )
    if dataset == "Reuters":
        fetcher = disk.PredefinedTextFileDatasetFetcher(
            random_state=random_state,
            dataset="Reuters",
            categories=[
                "acq",
                "corn",
                "crude",
                "earn",
                "grain",
                "interest",
                "money-fx",
                "ship",
                "trade",
                "wheat",
            ],
        )
    if dataset == "WebKB":
        fetcher = disk.RandomizedTextFileDatasetFetcher(
            random_state=random_state,
            dataset="WebKB",
            categories=["student", "faculty", "course", "project"],
        )
    # Datasets acquired from huggingface
    if dataset == "ag_news":
        fetcher = huggingface.PredefinedClassificationFetcher(
            random_state=random_state,
            path="ag_news",
        )
    if dataset == "amazon_polarity":
        fetcher = huggingface.PredefinedClassificationFetcher(
            random_state=random_state,
            path="amazon_polarity",
        )
    if dataset == "emotion":
        fetcher = huggingface.PredefinedClassificationFetcher(
            random_state=random_state,
            path="emotion",
        )
    if dataset == "glue":
        fetcher = huggingface.RandomizedClassificationFetcher(
            random_state=random_state,
            path="glue",
            name="sst2",
        )
    if dataset == "imdb":
        fetcher = huggingface.PredefinedClassificationFetcher(
            random_state=random_state,
            path="imdb",
        )
    if dataset == "rotten_tomatoes":
        fetcher = huggingface.PredefinedClassificationFetcher(
            random_state=random_state,
            path="rotten_tomatoes",
        )
    if dataset == "tweet_eval":
        fetcher = huggingface.PredefinedClassificationFetcher(
            random_state=random_state,
            path="tweet_eval",
            name="emotion",
        )
    # Datasets acquired from scikit-learn
    if dataset == "20NewsGroups-singlelabel":
        fetcher = scikit_learn.ScikitLearnDatasetFetcher(
            random_state=random_state,
            dataset="20NewsGroups",
        )
    if dataset == "Covertype":
        fetcher = scikit_learn.ScikitLearnDatasetFetcher(
            random_state=random_state,
            dataset="Covertype",
        )
    if dataset == "Iris":
        fetcher = scikit_learn.ScikitLearnDatasetFetcher(
            random_state=random_state,
            dataset="Iris",
        )

    if stream:
        X_train, X_test, y_train, y_test, target_names = fetcher.stream()
    else:
        X_train, X_test, y_train, y_test, target_names = fetcher.fetch()

    return X_train, X_test, y_train, y_test, target_names
