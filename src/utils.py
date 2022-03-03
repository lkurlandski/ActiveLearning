"""Random useful things unrelated to active learning, machine learning, or even mathematics.
"""

import inspect
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Iterator


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
