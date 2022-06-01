"""Get training and test data from files on disk.
"""

from collections import OrderedDict
import json
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.io import mmread
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle

from active_learning import utils


valid_disk_datasets = {
    "20newsgroups-multilabel",
    "avila",
    "rcv1_v2",
    "reuters",
    "web_kb",
}


preprocessed_datatsets = {"avila"}


datasets_path = Path("/projects/nlp-ml/io/input/ActiveLearningDatasets/")


train_path = "train"
test_path = "test"
feature_file = "X.mtx"  # .mtx extension supported
target_file_stem = "y"  # .mtx and .npy extensions supported


def paths_to_contents_generator(paths: Iterable[str]) -> Generator[str, None, None]:
    """Convert a corpus of filenames and to a generator of documents.

    Parameters
    ----------
    paths : Iterable[str]
        Filenames to read and return as generator.

    Returns
    -------
    Generator[str, None, None]
        A generator of documents.
    """

    return (open(f, "rb").read().decode("utf8", "replace") for f in paths)


def change_representation_if_mulitlabel(
    X: Iterable[str], y: Iterable[Any]
) -> Tuple[Iterable[str], Union[Iterable[Any], sparse.csr_matrix]]:
    """Detect duplicate files with different labels and restructure to support multilabel.

    Parameters
    ----------
    X : Iterable[str]
        Paths to documents of text.
    y : Iterable[Any]
        Corresponding labels for the files.

    Returns
    -------
    Tuple[Iterable[str], Union[Iterable[Any], sparse.csr_matrix]]
        The original data if no duplicate files were detected; else the data restructured to
            support multilabel classification.
    """

    # Return original data if no documents are represented multiple times under different labels
    if len(np.unique([Path(x).name for x in X])) == len(X):
        return X, y

    # Map between document name and the labels associated with the document
    tracker: Dict[str, List] = OrderedDict()
    for path, label in zip(X, y):
        name = Path(path).name
        if name not in tracker:
            tracker[name] = [label]
        else:
            tracker[name].append(label)

    # Rebuild the document-label arrays, using full document path
    X_, y_ = [], []
    accounted = set()
    for x in map(Path, X):
        if x.name not in accounted:
            accounted.add(x.name)
            X_.append(x.as_posix())
            y_.append(tracker[x.name])

    # Convert target into one-hot encoded 2D array
    mlb = MultiLabelBinarizer(sparse_output=True)
    y_ = mlb.fit_transform(y_)

    return X_, y_


def fetch_preprocessed(
    dataset: str, *, test_size: Optional[float] = None, random_state: Optional[int] = None
) -> Tuple[
    Union[np.array, sparse.csr_matrix],
    Union[np.array, sparse.csr_matrix],
    np.ndarray,
    np.ndarray,
    Dict[str, int],
]:

    path = datasets_path / dataset

    has_split = (path / train_path).exists() and (path / test_path).exists()
    if not has_split and test_size is None:
        raise ValueError(
            f"No test set found for {dataset} in {path.as_posix()}. "
            "You need to specify a test_size to split the dataset."
        )

    if has_split:
        X_train = mmread(path / train_path / feature_file)
        X_test = mmread(path / test_path / feature_file)
        y_train = utils.read_array_file_unknown_extension(path / train_path / target_file_stem)
        y_test = utils.read_array_file_unknown_extension(path / test_path / target_file_stem)

    else:
        X = mmread(path / feature_file)
        y = utils.read_array_file_unknown_extension(path / target_file_stem)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,
        )

    with open(path / "target_names.json", "r", encoding="utf8") as f:
        target_names = json.load(f)

    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

    return X_train, X_test, y_train, y_test, target_names


def stream(
    dataset: str,
    *,
    test_size: Optional[float] = None,
    categories: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[
    Generator[str, None, None], Generator[str, None, None], np.ndarray, np.ndarray, Dict[str, int]
]:

    path = datasets_path / dataset

    has_split = (path / train_path).exists() and (path / test_path).exists()
    if not has_split and test_size is None:
        raise ValueError(
            f"No test set found for {dataset} in {path.as_posix()}. "
            "You need to specify a test_size to split the dataset."
        )

    if has_split:
        bunch = load_files(
            path / train_path,
            categories=categories,
            load_content=False,
            random_state=random_state,
        )
        X_train, y_train, names_train = bunch["filenames"], bunch["target"], bunch["target_names"]
        X_train, y_train = change_representation_if_mulitlabel(X_train, y_train)

        bunch = load_files(
            path / test_path,
            categories=categories,
            load_content=False,
            random_state=random_state,
        )
        X_test, y_test, names_test = bunch["filenames"], bunch["target"], bunch["target_names"]
        X_test, y_test = change_representation_if_mulitlabel(X_test, y_test)

        if names_train != names_test:
            raise Exception(
                "Expected the target_names to match for the train and test sets."
                f"\nnames_train: {names_train}\nnames_test: {names_test}"
            )

        target_names = names_train

    else:
        bunch = load_files(
            path,
            categories=categories,
            load_content=False,
            random_state=random_state,
        )
        X, y, target_names = bunch["filenames"], bunch["target"], bunch["target_names"]
        X, y = change_representation_if_mulitlabel(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    X_train = paths_to_contents_generator(X_train)
    X_test = paths_to_contents_generator(X_test)

    target_names = {t: i for i, t in enumerate(target_names)}

    return X_train, X_test, y_train, y_test, target_names


def fetch(
    dataset: str,
    *,
    test_size: Optional[float] = None,
    categories: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Dict[str, int]]:

    X_train, X_test, y_train, y_test, target_names = stream(
        dataset, test_size=test_size, categories=categories, random_state=random_state
    )

    return list(X_train), list(X_test), y_train, y_test, target_names


def get_dataset(
    dataset: str,
    streaming: bool,
    *,
    test_size: Optional[Union[int, float]] = None,
    categories: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[Iterable[str], Iterable[str], np.ndarray, np.ndarray, Dict[str, int]]:

    if dataset in preprocessed_datatsets:
        return fetch_preprocessed(dataset, test_size=test_size, random_state=random_state)
    if streaming:
        return stream(
            dataset, test_size=test_size, categories=categories, random_state=random_state
        )
    return fetch(dataset, test_size=test_size, categories=categories, random_state=random_state)
