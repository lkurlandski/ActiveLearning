"""Get training and test data.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from active_learning.dataset_fetchers import disk, scikit_learn, huggingface


valid_datasets = set().union(
    disk.valid_disk_datasets,
    huggingface.valid_huggingface_datasets,
    scikit_learn.valid_scikit_learn_datasets,
)


def get_dataset(
    dataset: str, streaming: bool, *, random_state: Optional[int] = None
) -> Tuple[Iterable[Any], Iterable[Any], np.ndarray, np.ndarray, np.ndarray]:

    # Datasets extracted from disk
    if dataset == "20newsgroups-multilabel":
        X_train, X_test, y_train, y_test, target_names = disk.get_dataset(
            "20newsgroups-multilabel", streaming, random_state=random_state
        )
    elif dataset == "avila":
        X_train, X_test, y_train, y_test, target_names = disk.get_dataset(
            "avila", streaming, random_state=random_state
        )
    elif dataset == "rcv1_v2":
        X_train, X_test, y_train, y_test, target_names = disk.get_dataset(
            "rcv1_v2",
            streaming,
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
            random_state=random_state,
        )
    elif dataset == "reuters":
        X_train, X_test, y_train, y_test, target_names = disk.get_dataset(
            "reuters",
            streaming,
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
            random_state=random_state,
        )
    elif dataset == "web_kb":
        X_train, X_test, y_train, y_test, target_names = disk.get_dataset(
            dataset,
            streaming,
            test_size=0.2,
            categories=["student", "faculty", "course", "project"],
            random_state=random_state,
        )

    # Datasets acquired from huggingface.
    elif dataset in {"ag_news", "amazon_polarity", "emotion", "imdb", "rotten_tomatoes"}:
        X_train, X_test, y_train, y_test, target_names = huggingface.get_dataset(
            dataset, streaming, random_state=random_state
        )
    elif dataset == "glue-sst2":
        X_train, X_test, y_train, y_test, target_names = huggingface.get_dataset(
            "glue", streaming, random_state=random_state, test_size=0.2, name="sst2"
        )
    elif dataset == "tweet_eval":
        X_train, X_test, y_train, y_test, target_names = huggingface.get_dataset(
            dataset, streaming, random_state=random_state, name="emotion"
        )

    # Datasets acquired from scikit-learn.
    elif dataset == "20newsgroups-singlelabel":
        X_train, X_test, y_train, y_test, target_names = scikit_learn.get_dataset(
            "20newsgroups-singlelabel", streaming, random_state=random_state
        )
    elif dataset == "20newsgroups-singlelabel-vectorized":
        X_train, X_test, y_train, y_test, target_names = scikit_learn.get_dataset(
            "20newsgroups-singlelabel", streaming, random_state=random_state
        )
    elif dataset == "covertype":
        X_train, X_test, y_train, y_test, target_names = scikit_learn.get_dataset(
            "covertype", streaming, random_state=random_state, test_size=0.2
        )
    elif dataset == "iris":
        X_train, X_test, y_train, y_test, target_names = scikit_learn.get_dataset(
            "iris", streaming, random_state=random_state, test_size=0.2
        )
    elif dataset == "rcv1_v2-vectorized":
        X_train, X_test, y_train, y_test, target_names = scikit_learn.get_dataset(
            "iris", streaming, random_state=random_state
        )

    return X_train, X_test, y_train, y_test, target_names
