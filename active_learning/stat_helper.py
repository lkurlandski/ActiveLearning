"""Helpful statistical functions to assist in math and data wrangling.
"""

from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse


def mean_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Return the ``mean'' of a dataframe, taken in an element-wise fashion.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        Dataframes to operate upon.

    Returns
    -------
    pd.DataFrame
        Meaned dataframe.
    """
    mean_df = (
        pd.concat(dfs, keys=list(range(len(dfs))))  # concat with specific keys
        .groupby(level=1)  # group on first level
        .mean()  # take mean
        .reindex(dfs[0].index)  # use index order of first arg
    )

    return mean_df


def mean_dataframes_from_files(paths: List[Path], **read_csv_kwargs) -> pd.DataFrame:
    """Return the ``mean'' of a dataframe, taken in an element-wise fashion, from csv files.

    Parameters
    ----------
    paths : List[Path]
        .csv files to be read as dataframes

    Other Parameters
    ----------------
    **read_csv_kwargs : dict
        Named arguments passed along to pd.read_csv()

    Returns
    -------
    pd.DataFrame
        Meaned dataframe.
    """
    dfs = [pd.read_csv(p, **read_csv_kwargs) for p in paths]
    return mean_dataframes(dfs)


def delete_rows_csr(mat: sparse.csr_matrix, idx: Sequence[int]) -> sparse.csr_matrix:
    """Delete selected entries from a sparse matrix.

    Parameters
    ----------
    mat : sparse.csr_matrix
        A sparse matrix to delete rows from.
    idx : Sequence[int]
        Rows from the matrix to remove.

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix with selected rows removed.

    Raises
    ------
    ValueError
        If the passed matrix is not a csr_matrix.
    """
    if not sparse.isspmatrix_csr(mat):
        raise ValueError(f"Works only for CSR format, not {type(mat)}")

    idx = list(idx)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[idx] = False
    return mat[mask]


def remove_ids_from_array(
    a: Union[np.ndarray, sparse.csr_matrix], idx: np.ndarray
) -> Union[np.ndarray, sparse.csr_matrix]:
    """Remove rows from one of several types of arrays.

    Parameters
    ----------
    a : Union[np.ndarray, sparse.csr_matrix]
        An array to remove rows from.
    idx : np.ndarray
       Rows the array to remove.

    Returns
    -------
    Union[np.ndarray, sparse.csr_matrix]
        The modified array, of same type as input.

    Raises
    ------
    ValueError
        If the type of input array is not supported.
    """
    if isinstance(a, np.ndarray):
        return np.delete(a, idx, axis=0)
    if isinstance(a, sparse.csr_matrix):
        return delete_rows_csr(a, idx)

    raise TypeError(f"Unsupported type: {type(a)}")


def iter_spmatrix(matrix: sparse.spmatrix):
    """Iterator for iterating the elements in a sparse.*_matrix.

    This will always return:
    >>> (row, column, matrix-element)

    Currently this can iterate `coo`, `csc`, `lil` and `csr`, others may easily be added.

    Parameters
    ----------
    matrix : sparse.spmatrix
        The sparse matrix to iterate non-zero elements
    """
    if sparse.isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif sparse.isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c + 1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif sparse.isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r + 1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif sparse.isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")


def multi_argmin(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances lowest values.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.

    Returns:
        The indices of the n_instances smallest values.
    """
    assert (
        n_instances <= values.shape[0]
    ), "n_instances must be less or equal than the size of utility"

    max_idx = np.argpartition(values, n_instances - 1, axis=0)[:n_instances]
    return max_idx
