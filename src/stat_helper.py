"""Helpful statistical functions to assist in math and data wrangling.
"""

from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, List

import numpy as np
import pandas as pd
import scipy

# TODO: if a column of the original dataframe was integer type and the corresponding column in
# the new dataframe is also integer, but represented by a floating point (eg 1.000),
# change the new column's type to the same type as the orignal column
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

    # Coherce certain types of the mean dataframe to be same as the original dataframe, if possible
    # for c in mean_df.columns.tolist():
    #    org_dtype = dfs[0][c].dtype
    #    new_dtype = mean_df[c].dtype

    #    if pd.api.types.is_integer_dtype(org_dtype) and pd.api.types.is_float_dtype(new_dtype):
    #        if all(i.is_integer() for i in mean_df[c]):
    #            mean_df[c] = mean_df[c].as_type(int)

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


# TODO: test and document this function
def delete_rows_csr(mat, idx):

    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError(f"Works only for CSR format, not {type(mat)}")

    idx = list(idx)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[idx] = False
    return mat[mask]


# TODO: test and document this function
def delete_row_csr(mat, i):

    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError(f"Works only for CSR format, not {type(mat)}")

    if not isinstance(i, int):
        raise ValueError(f"Works only for a single integer index, not {type(i)}")

    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i] : -n] = mat.data[mat.indptr[i + 1] :]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i] : -n] = mat.indices[mat.indptr[i + 1] :]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i + 1 :]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

    return mat


# TODO: test and document this function
def delete_row_lil(mat, i):

    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError(f"Works only for LIL format, not {type(mat)}")

    if not isinstance(i, int):
        raise ValueError(f"Works only for a single integer index, not {type(i)}")

    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

    return mat


# TODO: test and document this function
def remove_ids_from_array(a: Any, idx: np.ndarray) -> Any:
    """Remove certain elements from several types of contiguous data structures.

    Parameters
    ----------
    a : Any
        An input array or data structure
    idx : np.ndarray
        ids to remove from the input data structure

    Returns
    -------
    Any
        The input array with the seleted ids removed (same type as input)

    Raises
    ------
    ValueError
        If the type of the input array is unexpected
    """

    if isinstance(a, np.ndarray):
        return np.delete(a, idx, axis=0)
    if isinstance(a, scipy.sparse.csr_matrix):
        return delete_rows_csr(a, idx)
    if isinstance(a, scipy.sparse.lil_matrix):
        return delete_row_lil(a, idx)

    raise ValueError(f"Unknown matrix passed to remove_ids_from_array: {type(a)}")


if __name__ == "__main__":
    pass
