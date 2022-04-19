"""Helper functions for the datasets module.
"""

from typing import Generator, Iterable


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
