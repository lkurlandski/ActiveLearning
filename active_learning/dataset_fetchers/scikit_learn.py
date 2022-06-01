"""Fetch datasets using the scikit-learn API.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_20newsgroups_vectorized,
    fetch_covtype,
    fetch_rcv1,
    load_iris,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


valid_scikit_learn_datasets = {
    "20newsgroups-singlelabel",
    "20newsgroups-singlelabel-vectorized",
    "covertype",
    "iris",
    "rcv1_v2-vectorized",
}

preprocessed_datatsets = {
    "20newsgroups-singlelabel-vectorized",
    "covertype",
    "iris",
    "rcv1_v2-vectorized",
}


def stream(
    dataset: str,
    *,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
) -> Tuple[
    Generator[str, None, None],
    Generator[str, None, None],
    np.ndarray,
    np.ndarray,
    Dict[str, int],
]:

    warnings.warn("WARNING: Only pseudo-streaming is possible with scikit-learn datasets.")

    X_train, X_test, y_train, y_test, target_names = fetch(
        dataset, test_size=test_size, random_state=random_state
    )
    return (x for x in X_train), (x for x in X_test), y_train, y_test, target_names


def fetch(
    dataset: str,
    *,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
) -> Tuple[
    Union[List[str], np.ndarray],
    Union[List[str], np.ndarray],
    np.ndarray,
    np.ndarray,
    Dict[str, int],
]:

    if dataset == "20newsgroups-singlelabel":
        bunch = fetch_20newsgroups(
            subset="train",
            remove=("headers", "footers", "quotes"),
            random_state=random_state,
            shuffle=True,
        )
        X_train, y_train = bunch["data"], bunch["target"]
        bunch = fetch_20newsgroups(
            subset="test",
            remove=("headers", "footers", "quotes"),
            random_state=random_state,
            shuffle=True,
        )
        X_test, y_test = bunch["data"], bunch["target"]

    elif dataset == "20newsgroups-singlelabel-vectorized":
        bunch = fetch_20newsgroups_vectorized(
            subset="train", remove=("headers", "footers", "quotes")
        )
        X_train, y_train = shuffle(bunch["data"], bunch["target"])
        bunch = fetch_20newsgroups_vectorized(
            subset="test", remove=("headers", "footers", "quotes")
        )
        X_test, y_test = shuffle(bunch["data"], bunch["target"])

    if dataset == "covertype":
        bunch = fetch_covtype(random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            bunch["data"], bunch["target"], test_size=test_size, random_state=random_state
        )

    elif dataset == "iris":
        bunch = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            bunch["data"], bunch["target"], test_size=test_size, random_state=random_state
        )

    elif dataset == "rcv1_v2-vectorized":
        bunch = fetch_rcv1(subset="train", random_state=random_state, shuffle=True)
        X_train, y_train = bunch["data"], bunch["target"]
        bunch = fetch_rcv1(subset="test", random_state=random_state, shuffle=True)
        X_test, y_test = bunch["data"][: y_train.shape[0]], bunch["target"][: y_train.shape[0]]
        print(
            f"Randomly selected {y_train.shape[0]} samples from the test set of rcv1_v2 to use "
            "as a test set instead of the full 781265 samples."
        )

    target_names = {t: i for i, t in enumerate(bunch["target_names"])}

    return X_train, X_test, y_train, y_test, target_names


def get_dataset(
    dataset: str,
    streaming: bool,
    *,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
) -> Tuple[
    Union[Iterable[str], np.ndarray],
    Union[Iterable[str], np.ndarray],
    np.ndarray,
    np.ndarray,
    Dict[str, int],
]:

    if dataset not in valid_scikit_learn_datasets:
        raise ValueError(
            f"Dataset {dataset} not recognized. "
            f"Recognized scikit-learn datasets are {valid_scikit_learn_datasets}."
        )

    if dataset in preprocessed_datatsets:
        X_train, X_test, y_train, y_test, target_names = fetch(
            dataset, test_size=test_size, random_state=random_state
        )
    elif streaming:
        X_train, X_test, y_train, y_test, target_names = stream(
            dataset, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test, target_names = fetch(
            dataset, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test, target_names
