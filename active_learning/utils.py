"""Random useful things unrelated to active learning, machine learning, or even mathematics.
"""

import inspect
from itertools import zip_longest
import math
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Generator, Iterator, Iterable, List, Tuple, Union

import numpy as np
import psutil
from scipy import sparse
from scipy.io import mmread


def tree(
    dir_path: Path,
    prefix: str = "",
    space: str = "    ",
    branch: str = "│   ",
    tee: str = "├── ",
    last: str = "└── ",
) -> Iterator[str]:
    """Return a unix tree like representation of a path.

    Parameters
    ----------
    dir_path : Path
        Path to print structure of
    prefix : str, optional
        formatting, by default ''
    space : str, optional
        formatting, by default '    '
    branch : str, optional
        formatting, by default '│   '
    tee : str, optional
        formatting, by default '├── '
    last : str, optional
        formatting, by default '└── '

    Yields
    ------
    Iterator[str]
        Each element of the tree-like output
    """

    contents = list(dir_path.iterdir())
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():
            extension = branch if pointer == tee else space
            yield from tree(path, prefix=prefix + extension)


def check_callable_has_parameter(_callable: Callable[..., Any], parameter: str) -> bool:
    """Determine if a callable object, such as a function or class, has a particular parameter.

    Parameters
    ----------
    callable : Callable[..., Any]
        Callable object, e.g., a function
    parameter : str
        parameter to check for the presence of

    Returns
    -------
    bool
        If the paramater is present or not
    """

    argspec = inspect.getfullargspec(_callable)
    args = set(argspec.args + argspec.kwonlyargs)
    if parameter in args:
        return True
    return False


def format_bytes(n_bytes: int) -> str:
    """Return a string representation of an amount of bytes.

    Parameters
    ----------
    bytes : int
        Number of bytes

    Returns
    -------
    str
        String representation, including a unit such as MB
    """

    if n_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(n_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(n_bytes / p, 2)
    return f"{s}{size_name[i]}"


def print_memory_stats(flush: bool = True, attrs: List[str] = None) -> str:
    """Print statistics about the usage of memory.

    Parameters
    ----------
    flush : bool
        Passed as argument to flush in print()
    attrs : List[str]
        Attributes of memory to print, any of
            {
                "total", "available", "percent", "used", "free", "active",
                "inactive", "buffers", "cached", "shared", "slab"
            }
            Defaults to None, which will print all attributes of memory

    Returns
    -------
    str
        The formatted memory information
    """

    attrs = (
        [
            "total",
            "available",
            "percent",
            "used",
            "free",
            "active",
            "inactive",
            "buffers",
            "cached",
            "shared",
            "slab",
        ]
        if attrs is None
        else attrs
    )

    mem = psutil.virtual_memory()
    s = "Memory\n------"
    for a in attrs:
        data = format_bytes(getattr(mem, a)) if a != "percent" else f"{getattr(mem, a)}%"
        s += f"\n\t{a}={data}"
    print(s, flush=flush)

    return s


def nbytes(a: Any) -> int:
    """Bet the number of bytes in one of several different types of data structures.

    Parameters
    ----------
    a : Any
        Data structure to determine number of bytes in

    Returns
    -------
    int
        Number of bytes in the array

    Raises
    ------
    ValueError
        If the type of object is not supported
    """

    if isinstance(a, np.ndarray):
        return a.nbytes
    if sparse.issparse(a):
        return a.data.nbytes + a.indptr.nbytes + a.indices.nbytes

    raise ValueError(f"Type not recognized: {type(a)}")


def grouper(
    iterable: Iterable[Any], chunk_size: int, fill_value: Any = None
) -> Generator[Tuple[Any], None, None]:
    """Return the elements of an interable in batches.

    Parameters
    ----------
    iterable : Iterable[Any]
        Iterable to return elements from
    chunk_size : int
        Number of elements in each batch
    fill_value : Any, optional
        Filler for potentially empty elements of the final batch

    Returns
    ------
    Generator[Tuple[Any], None, None]
        Tuples of length=chunk_size that contains the elements from the iterable
    """

    args = [iter(iterable)] * chunk_size

    return zip_longest(*args, fillvalue=fill_value)


def get_array_file_ext(a: Union[sparse.csr_matrix, np.ndarray]) -> str:
    """Get the correct file extension for saving an array.

    Parameters
    ----------
    a : Union[sp.csr_matrix, np.ndarray]
        The array to analyze and determine the correct extension for.

    Returns
    -------
    str
        The extension, either .mtx or .npy.

    Raises
    ------
    ValueError
        If the number of dimensions is not 1 or 2.
    """
    if a.ndim == 1:
        return ".npy"
    if a.ndim == 2:
        return ".mtx"

    raise ValueError(f"Expected a one or two dimensional matrix, not {a.ndim} dimensional.")


def read_array_file(file: Union[str, Path], **kwargs) -> Union[np.ndarray, sparse.csr_matrix]:
    """Read a sparse/dense ndimensional array file.

    Parameters
    ----------
    file : Union[str, Path]
        _description_

    Returns
    -------
    Union[np.ndarray, sparse.csr_matrix]
        The array read from the file.

    Raises
    ------
    ValueError
        If an extension other than '.mtx', '.npy', and '.npz' is supplied.
    """
    file = Path(file)
    if file.suffix == ".mtx":
        return mmread(file, **kwargs)
    if file.suffix in (".npy", ".npz"):
        return np.load(file, **kwargs)
    raise ValueError(
        f"Unrecognized file extension: '{file.suffix}'. "
        f"Supported extensions are: '.mtx', '.npy', and '.npz'."
    )


def read_array_file_unknown_extension(
    file_without_extension: Union[str, Path], **kwargs
) -> Union[np.ndarray, sparse.csr_matrix]:
    """Read an array from disk given the name of the file without the extension.

    Parameters
    ----------
    file_without_extension : Union[str, Path]
        A file name without the extension, e.g.,
        `path/to/some/array`, but not `path/to/some/array.mtx`. Capable of reading ".mtx", ".npy",
        and ".npz" files.

    Returns
    -------
    Union[np.ndarray, sparse.csr_matrix]
        The array read from disk.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    """
    file = Path(file_without_extension)
    for ext in (".mtx", ".npy", ".npz"):
        try:
            return read_array_file(file.with_suffix(ext), **kwargs)
        except FileNotFoundError:
            pass

    raise FileNotFoundError(
        f"Cannot find {file} with any of the extensions: '.mtx', '.npy', and '.npz'."
    )
