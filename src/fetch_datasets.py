"""Get training and test data.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Tuple

from datasets import load_dataset
import numpy as np
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_covtype,
    load_iris,
)
from sklearn.model_selection import train_test_split

import config
import stat_helper


def get_covertype(
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Covertype dataset.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """

    bunch = fetch_covtype(random_state=random_state, shuffle=True)

    X = bunch["data"]
    y = bunch["target"]
    target_names = np.array(
        [1, 2, 3, 4, 5, 6, 7]
    )  # TODO: once scikit-learn fixes bug, use bunch['target_names']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, target_names


def get_20_newsgroups(
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 20NewsGroups dataset in a raw format.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """

    bunch = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    X_train = np.array(bunch["data"])
    y_train = np.array(bunch["target"])
    X_train, y_train = stat_helper.shuffle_corresponding_arrays(X_train, y_train, random_state)

    bunch = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    X_test = np.array(bunch["data"])
    y_test = np.array(bunch["target"])
    target_names = np.array(bunch["target_names"])

    return X_train, X_test, y_train, y_test, target_names


def get_iris(
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Iris dataset.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """

    bunch = load_iris()
    X = np.array(bunch["data"])
    y = np.array(bunch["target"])
    target_names = np.array(bunch["target_names"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, target_names


def get_avila(
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Avila dataset.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """

    avila_root = config.dataset_paths["Avila"]

    X_train = np.loadtxt(avila_root / "train/X.csv", delimiter=",")
    y_train = np.loadtxt(avila_root / "train/y.csv", dtype=np.str_)
    X_test = np.loadtxt(avila_root / "test/X.csv", delimiter=",")
    y_test = np.loadtxt(avila_root / "test/y.csv", dtype=np.str_)
    X_train, y_train = stat_helper.shuffle_corresponding_arrays(X_train, y_train, random_state)

    target_names = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"])

    return X_train, X_test, y_train, y_test, target_names

def get_glue_cola(
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the cola subset of glue dataset.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """
    
    bunch = load_dataset('glue', 'cola')

    X_train = np.array(bunch['train']['sentence'])
    y_train = np.array(bunch['train']['label'])

    X_train, y_train = stat_helper.shuffle_corresponding_arrays(X_train, y_train, random_state)

    X_test = np.array(bunch['test']['sentence'])
    y_test = np.array(bunch['test']['label'])

    target_names = np.array([0,1])

    return X_train, X_test, y_train, y_test, target_names

def get_glue_sst2(
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the sst2 subset of glue dataset.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """
    
    bunch = load_dataset('glue', 'sst2')

    X_train = np.array(bunch['train']['sentence'])
    y_train = np.array(bunch['train']['label'])

    X_train, y_train = stat_helper.shuffle_corresponding_arrays(X_train, y_train, random_state)

    X_test = np.array(bunch['test']['sentence'])
    y_test = np.array(bunch['test']['label'])

    target_names = np.array([0,1])

    return X_train, X_test, y_train, y_test, target_names

# TODO: attempt to access bunch['file_names'] and create a streaming approach for text datasets
# TODO: design a better way of handling the target_names using a LabelEncoder
def get_dataset(
    dataset: str,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return any implemented dataset.

    Parameters
    ----------
    dataset : str
        Code to refer to a particular dataset
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

    if dataset == "Avila":
        X_train, X_test, y_train, y_test, target_names = get_avila(random_state)
    elif dataset == "20NewsGroups":
        X_train, X_test, y_train, y_test, target_names = get_20_newsgroups(random_state)
    elif dataset == "Covertype":
        X_train, X_test, y_train, y_test, target_names = get_covertype(random_state)
    elif dataset == "Iris":
        X_train, X_test, y_train, y_test, target_names = get_iris(random_state)
    elif dataset == "Glue_cola":
        X_train, X_test, y_train, y_test, target_names = get_glue_cola(random_state)
    elif dataset == "Glue_sst2":
        X_train, X_test, y_train, y_test, target_names = get_glue_sst2(random_state)
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")

    return X_train, X_test, y_train, y_test, target_names


def test() -> None:
    """Test."""

    for dataset in ("Avila", "20NewsGroups", "Covertype", "Iris", "Glue_cola", "Glue_sst2"):
        print(f"Fetching {dataset}")
        X_train, X_test, y_train, y_test, target_names = get_dataset(dataset, 0)
        for d in (X_train, X_test, y_train, y_test, target_names):
            print(f"{type(d)} -- {d.shape}")
    print("Done.")


if __name__ == "__main__":
    test()
