"""Get training and test data.
"""

from typing import Tuple

import numpy as np

import fetch_datasets
import feature_extraction
import utils


def get_data(
    dataset: str, feature_representation: str = None, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a fully processed dataset, ready to be used to train and test a model.

    Parameters
    ----------
    dataset : str
        Code to refer to a particular dataset
    feature_representation : str
        Code to refer to a particular method of representing the dataset, defaults to None
    random_state : int, optional
        Integer used for reproducible randomization, defaults to 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories

    Raises
    ------
    ValueError
        If the dataset code is not recognized
    """

    print("Loading Raw Datasets.")
    utils.print_memory_stats(True)
    X_train, X_test, y_train, y_test, labels = fetch_datasets.get_dataset(dataset, random_state)

    print("Performing Feature Extraction.")
    utils.print_memory_stats(True)
    X_train, X_test = feature_extraction.get_features(X_train, X_test, feature_representation)

    print("Returning Processed Data.")
    utils.print_memory_stats(True)
    return X_train, X_test, y_train, y_test, labels
