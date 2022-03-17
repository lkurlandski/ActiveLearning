"""Get training and test data.
"""

from typing import Tuple

import numpy as np

import fetch_datasets
import feature_extraction


def get_data(
    dataset: str, stream: bool, feature_representation: str, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a fully processed dataset, ready to be used to train and test a model.

    Parameters
    ----------
    dataset : str
        Code to refer to a particular dataset
    stream : bool
        Controls whether data is streamed or returned in full
    feature_representation : str
        Code to refer to a particular method of representing the dataset, defaults to None
    random_state : int, optional
        Integer used for reproducible randomization, defaults to 0

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The arrays for X_train, y_train, X_test, and y_test, along with the set of categories
    """

    print(f"Loading Raw Datasets; stream={stream}.")
    X_train, X_test, y_train, y_test, labels = fetch_datasets.get_dataset(
        dataset, stream, random_state
    )

    print(f"Performing Feature Extraction; stream={stream}.")
    X_train, X_test = feature_extraction.get_features(
        X_train, X_test, feature_representation, stream
    )

    print("Returning Processed Data.")

    return X_train, X_test, y_train, y_test, labels
