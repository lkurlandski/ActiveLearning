"""Get training and test data.
"""

from pprint import pprint                                           # pylint: disable=unused-import
from pathlib import Path
import sys                                                          # pylint: disable=unused-import
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized, fetch_covtype, load_iris
from sklearn.model_selection import train_test_split

def shuffle_corresponding_arrays(
        a1:np.ndarray,
        a2:np.ndarray,
        random_state:int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle two arrays with the exact same ordering applied to each array.

    Parameters
    ----------
    a1 : np.ndarray
        First array
    a2 : np.ndarray
        Second array
    random_state : int
        _description_

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Integer used for reproducible randomization

    Raises
    ------
    ValueError
        If the arrays have different shapes along the first axis
    """

    if a1.shape[0] != a2.shape[0]:
        raise ValueError("Arrays are different lengths along first axis.")

    rng = np.random.default_rng(random_state)
    idx = np.arange(a1.shape[0])
    rng.shuffle(idx)

    return a1[idx], a2[idx]

def get_covtype(
        random_state:int
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

    X = bunch['data']
    y = bunch['target']
    labels = [1,2,3,4,5,6,7]    # bunch['target_names'] does not work

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=random_state)

    return X_train, X_test, y_train, y_test, labels

# TODO: use scipy.sparse.csr_array not scipy.sparse.csr_matrix (per the scipy.sparse docs)
def get_20_newsgroups(
        random_state:int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 20NewsGroups dataset in a vectorized format.

    Parameters
    ----------
    random_state : int
        Integer used for reproducible randomization

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """

    bunch = fetch_20newsgroups_vectorized(
        subset='train', remove=('headers', 'footers', 'quotes')
    )
    X_train = bunch['data']
    y_train = bunch['target']
    X_train, y_train = shuffle_corresponding_arrays(X_train, y_train, random_state)

    bunch = fetch_20newsgroups_vectorized(
        subset='test', remove=('headers', 'footers', 'quotes')
    )
    X_test = bunch['data']
    y_test = bunch['target']
    labels = list(bunch['target_names'])

    return X_train, X_test, y_train, y_test, labels

# TODO: rename this function
def get_20_newsgroups_bert(
        random_state:int
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

    bunch = fetch_20newsgroups(
        subset='train', remove=('headers', 'footers', 'quotes')
    )
    X_train = np.array(bunch['data'])
    y_train = np.array(bunch['target'])
    X_train, y_train = shuffle_corresponding_arrays(X_train, y_train, random_state)

    bunch = fetch_20newsgroups(
        subset='test', remove=('headers', 'footers', 'quotes')
    )
    X_test = np.array(bunch['data'])
    y_test = np.array(bunch['target'])
    labels = list(bunch['target_names'])

    return X_train, X_test, y_train, y_test, labels

def get_iris(
        random_state:int
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
    X = np.array(bunch['data'])
    y = np.array(bunch['target'])
    labels = bunch['target_names'].tolist()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=random_state)

    return X_train, X_test, y_train, y_test, labels

def get_avila(
        random_state:int
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

    avila_root = Path("/projects/nlp-ml/io/input/numeric/Avila")

    X_train = np.loadtxt(avila_root / "train/X.csv", delimiter=',')
    y_train = np.loadtxt(avila_root / "train/y.csv", dtype=np.str_)
    X_test = np.loadtxt(avila_root / "test/X.csv", delimiter=',')
    y_test = np.loadtxt(avila_root / "test/y.csv", dtype=np.str_)
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]

    if random_state is not None:
        X_train, y_train = shuffle_corresponding_arrays(X_train, y_train, random_state)
        X_test, y_test = shuffle_corresponding_arrays(X_train, y_train, random_state)

    return X_train, X_test, y_train, y_test, labels

# TODO: Establish proper data types and ensure proper data types are being used
# TODO: attempt to access bunch['file_names'] and create a streaming approach for text datasets
def get_dataset(
        dataset:str,
        random_state : int = None
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
        X_train, X_test, y_train, y_test, labels = get_avila(random_state)
    elif dataset == "20NewsGroups":
        X_train, X_test, y_train, y_test, labels = get_20_newsgroups(random_state)
    elif dataset == "20NewsGroups-raw":
        X_train, X_test, y_train, y_test, labels = get_20_newsgroups_bert(random_state)
    elif dataset == "Covertype":
        X_train, X_test, y_train, y_test, labels = get_covtype(random_state)
    elif dataset == "Iris":
        X_train, X_test, y_train, y_test, labels = get_iris(random_state)
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")

    return X_train, X_test, y_train, y_test, labels

def test():
    """Test.
    """
    X_train, y_train, X_test, y_test, labels = get_dataset("20NewsGroups-bert", 0)
    print(X_train, y_train, X_test, y_test, labels)

if __name__ == "__main__":
    test()
