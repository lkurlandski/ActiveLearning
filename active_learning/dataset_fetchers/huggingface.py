"""Fetch datasets using the huggingface API.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

from datasets import DatasetDict, load_dataset
import numpy as np


# Contains the name of the feature to use as data from each dataset
feature_keys = {
    "ag_news": "text",
    "amazon_polarity": "content",
    "emotion": "text",
    "glue": "sentence",
    "imdb": "text",
    "rotten_tomatoes": "text",
    "tweet_eval": "text",
}


valid_huggingface_datasets = set(feature_keys.keys())
valid_huggingface_datasets.remove("glue")
valid_huggingface_datasets.add("glue-sst2")


def stream(
    path: str,
    *,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[
    Generator[str, None, None], Generator[str, None, None], np.ndarray, np.ndarray, Dict[str, int]
]:

    # By disallowing an individual train/test/validation split, we simplify the return types
    if "split" in kwargs:
        raise ValueError(f"Loading an individual split is not supported: {kwargs['split']}.")

    dataset: DatasetDict = load_dataset(path, **kwargs)

    if "test" not in dataset and test_size is None:
        raise ValueError(
            f"No test set found in dataset {path}. "
            "You need to specify a test_size to split the dataset."
        )

    if test_size is not None:
        print(f"No default test split found. Using {test_size} of train split as test split.")
        dataset = dataset["train"].train_test_split(test_size=test_size, seed=random_state)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    X_train = (x[feature_keys[path]] for x in train_dataset)
    X_test = (x[feature_keys[path]] for x in test_dataset)

    y_train = np.array([x["label"] for x in train_dataset])
    y_test = np.array([x["label"] for x in test_dataset])

    target_names = {t: i for i, t in enumerate(train_dataset.info.features["label"].names)}

    return X_train, X_test, y_train, y_test, target_names


def fetch(
    path: str,
    *,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Dict[str, int]]:
    X_train, X_test, y_train, y_test, target_names = stream(
        path, test_size=test_size, random_state=random_state, **kwargs
    )

    return (
        list(X_train),
        list(X_test),
        np.array(list(y_train)),
        np.array(list(y_test)),
        target_names,
    )


def get_dataset(
    path: str,
    streaming: bool,
    *,
    test_size: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[Iterable[str], Iterable[str], np.ndarray, np.ndarray, Dict[str, int]]:
    if streaming:
        return stream(path, test_size=test_size, random_state=random_state, **kwargs)
    return fetch(path, test_size=test_size, random_state=random_state, **kwargs)
