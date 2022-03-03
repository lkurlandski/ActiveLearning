"""Random useful things unrelated to active learning, machine learning, or even mathematics.
"""

import math
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Iterator, Union


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


# Used to display a pathlib.Path object in a human-readable way.
# Copied from: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
class DisplayablePath:
    """Utility method for displaying a directory."""

    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path: Union[str, Path], parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)), key=lambda s: str(s).lower()
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))


def bytes_to_string(size_bytes: int) -> str:
    """Convert raw integer bytes into a nicely formatted string depending on the number of bytes.

    Parameters
    ----------
    size_bytes : int
        Number of bytes

    Returns
    -------
    str
        A formatted string, e.g., '24KB'
    """

    if size_bytes == 0:
        return "0B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return "%s %s" % (s, size_name[i])
