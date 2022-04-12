"""A basic encapsulation of a pool of data and its respective paths.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from scipy import sparse
from scipy.io import mmwrite, mmread


@dataclass
class Pool:
    """A simple encapsulation of a pool of data.

    Attributes
    ----------
    X : Union[np.ndarray, sparse.csr_matrix]
        The features of the data, defaults to None.
    y: Union[np.ndarray, sparse.csr_matrix]
        The labels of the data, defaults to None.
    X_path: Path
        A path associated with the features, defaults to None. Supported file extensions are
            .mtx and .npy.
    y_path: Path
        A path associated with the labels, defaults to None. Supported file extensions are
            .mtx and .npy.
    """

    X: Union[np.ndarray, sparse.csr_matrix] = None
    y: Union[np.ndarray, sparse.csr_matrix] = None
    X_path: Path = None
    y_path: Path = None

    def save(self) -> None:
        """Save the data to file.

        Raises
        ------
        Exception
            If X_path or y_path is None.
        Exception
            If X_path has an incompatible extension, and the protocol to save it is unknown.
        Exception
            If y_path has an incompatible extension, and the protocol to save it is unknown.
        """

        if self.X_path is None or self.y_path is None or self.X is None or self.y is None:
            raise Exception("One of the data items or the files associated with them is None.")

        if self.X.ndim == 2 and self.X_path.suffix == ".mtx":
            mmwrite(self.X_path, self.X)
        else:
            raise Exception(f"Extension {self.X_path.name} incompatible with ndim {self.X.ndim}.")

        if self.y.ndim == 2 and self.y_path.suffix == ".mtx":
            mmwrite(self.y_path, self.y)
        elif self.y.ndim == 1 and self.y_path.suffix == ".npy":
            np.save(self.y_path, self.y)
        else:
            raise Exception(f"Extension {self.y_path.name} incompatible with ndim {self.y.ndim}.")

    def load(self) -> Pool:
        """Load the data from a file.

        Returns
        -------
        Pool
            An instance of Pool with populated X and y data.

        Raises
        ------
        Exception
            If X_path or y_path is None.
        Exception
            If X_path has an incompatible extension, and the protocol to load it is unknown.
        Exception
            If y_path has an incompatible extension, and the protocol to load it is unknown.
        """

        if self.X_path is None or self.y_path is None:
            raise Exception("One of the data files is None.")

        if self.X_path.suffix == ".mtx":
            self.X = mmread(self.X_path)
            if sparse.isspmatrix_coo(self.X):
                self.X = sparse.csr_matrix(self.X)
        elif self.X_path.suffix == ".npy":
            self.X = np.load(self.X_path)
        else:
            raise Exception(f"Extension {self.X_path.name} incompatible with known read methods.")

        if self.y_path.suffix == ".mtx":
            self.y = mmread(self.y_path)
            if sparse.isspmatrix_coo(self.y):
                self.y = sparse.csr_matrix(self.y)
        elif self.y_path.suffix == ".npy":
            self.y = np.load(self.y_path)
        else:
            raise Exception(f"Extension {self.y_path.name} incompatible with known read methods.")

        return self
