"""Various tools to support encapsulation of data and logical processes associated with AL.
"""


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat, pprint  # pylint: disable=unused-import
import shutil
import sys  # pylint: disable=unused-import
from typing import List, Union

import numpy as np
from scipy import sparse
from scipy.io import mmwrite, mmread
from sklearn.utils import Bunch

from active_learning import utils
from active_learning.al_pipeline import (
    valid_early_stop_modes,
    valid_first_batch_modes,
)
from active_learning.dataset_fetchers import valid_datasets
from active_learning.estimators import valid_base_learners
from active_learning.feature_extractors import valid_feature_reps
from active_learning.query_strategies import valid_query_strategies


class Params(Bunch):
    """Encapsulate the parameters of the active learning experiment.

    Attributes
    ----------
    output_root : Path
        The root location to store experiment output.
    early_stop_mode : str
        The early stop mode to use.
    first_batch_mode : str
        The first batch mode to use.
    batch_size : int
        The batch size to use.
    query_strategy : str
        The query strategy to use.
    base_learner : str
        The base learner to use.
    feature_rep : str
        The feature representation to use, for extracting features from raw data.
    dataset : str
        The dataset to perform experiments with.
    random_state : int
        The random state to use, for reproducibility.
    """

    output_root: Path
    early_stop_mode: str
    first_batch_mode: str
    batch_size: int
    query_strategy: str
    base_learner: str
    feature_rep: str
    dataset: str
    random_state: int

    def __init__(
        self,
        output_root: Union[str, Path],
        early_stop_mode: str,
        first_batch_mode: str,
        batch_size: str,
        query_strategy: str,
        base_learner: str,
        feature_rep: str,
        dataset: str,
        random_state: Union[str, int, float],
        verify: bool = True,
    ) -> None:
        """Create the parameters object and optionally verify the parameters.

        Parameters
        ----------
        output_root : Path
            The root location to store experiment output.
        early_stop_mode : str
            The early stop mode to use.
        first_batch_mode : str
            The first batch mode to use.
        batch_size : int
            The batch size to use.
        query_strategy : str
            The query strategy to use.
        base_learner : str
            The base learner to use.
        feature_rep : str
            The feature representation to use, for extracting features from raw data.
        dataset : str
            The dataset to perform experiments with.
        random_state : int
            The random state to use, for reproducibility.
        verify : bool, optional
            If True, will verify to ensure the paramaters are valid. By default True.
        """
        super().__init__(
            output_root=output_root,
            early_stop_mode=early_stop_mode,
            first_batch_mode=first_batch_mode,
            batch_size=batch_size,
            query_strategy=query_strategy,
            base_learner=base_learner,
            feature_rep=feature_rep,
            dataset=dataset,
            random_state=random_state,
        )

        self.output_root = Path(output_root)
        self.early_stop_mode = early_stop_mode
        self.first_batch_mode = first_batch_mode
        self.batch_size = int(batch_size) if float(batch_size).is_integer() else float(batch_size)
        self.query_strategy = query_strategy
        self.base_learner = base_learner
        self.feature_rep = feature_rep
        self.dataset = dataset
        self.random_state = int(random_state)

        if verify:
            self.verify()

    def __str__(self) -> str:
        return f"""
            output_root: {self.output_root}
            early_stop_mode: {self.early_stop_mode}
            first_batch_mode: {self.first_batch_mode}
            batch_size: {self.batch_size}
            query_strategy: {self.query_strategy}
            base_learner: {self.base_learner}
            feature_rep: {self.feature_rep}
            dataset: {self.dataset}
            random_state: {self.random_state}
            """

    def construct_str(self) -> str:
        """Return a string that would construct this object if run with a python interpreter.

        Returns
        -------
        str
            A string to construct the object.
        """

        return (
            "Params("
            f"output_root='{self.output_root}', "
            f"early_stop_mode='{self.early_stop_mode}', "
            f"first_batch_mode='{self.first_batch_mode}', "
            f"batch_size={self.batch_size}, "
            f"query_strategy='{self.query_strategy}', "
            f"base_learner='{self.base_learner}', "
            f"feature_rep='{self.feature_rep}', "
            f"dataset='{self.dataset}', "
            f"random_state={self.random_state}, "
            "verify=True"
            ")"
        )

    def verify(self) -> None:
        """Verify that the supplied parameters are valid.

        Raises
        ------
        FileNotFoundError
            If the output_root does not exist.
        ValueError
            If any of the parameters are invalid.
        """
        if not self.output_root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.output_root.as_posix()}")

        if self.early_stop_mode not in valid_early_stop_modes:
            raise ValueError(
                f"Invalid early stop mode: {self.early_stop_mode}. "
                f"Valid modes are:\n{pformat(valid_early_stop_modes)}"
            )

        if self.first_batch_mode not in valid_first_batch_modes:
            raise ValueError(
                f"Invalid first batch mode: {self.first_batch_mode}. "
                f"Valid modes are:\n{pformat(valid_first_batch_modes)}"
            )

        if self.batch_size <= 0 or (self.batch_size > 1 and isinstance(self.batch_size, float)):
            raise ValueError(
                f"Invalid batch size: {self.batch_size}. Batch size should an integer greater than "
                "0 or a float between 0 and 1, exlcusive."
            )

        if self.query_strategy not in valid_query_strategies:
            raise ValueError(
                f"Invalid query strategy: {self.query_strategy}. "
                f"Valid strategies are:\n{pformat(valid_query_strategies)}"
            )

        if self.base_learner not in valid_base_learners:
            raise ValueError(
                f"Invalid base learner: {self.base_learner}. "
                f"Valid learners are:\n{pformat(valid_base_learners)}"
            )

        if self.feature_rep not in valid_feature_reps:
            raise ValueError(
                f"Invalid feature rep: {self.feature_rep}. "
                f"Valid feature reps:\n{pformat(valid_feature_reps)}"
            )

        if self.dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset: {self.dataset}. " f"Valid datsets:\n{pformat(valid_datasets)}"
            )


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


class OutputDataContainer:
    """Encapsulate the output files from a single AL experiment or an average across many.

    Attributes
    ----------
    test_set_str : str
        A string identifying the test set as it appears in the output structure.
    train_set_str : str
        A string identifying the training set as it appears in the output structure.
    subsets : Tuple[str]
        A tuple of strings identifying the subsets of data as they appear in the output structure.
    root : Path
        The root directory of the output data.
    training_data_file : Path
        The file containing the training data at each iteration.
    stopping_results_file : Path
        The file containing the stopping results.
    processed_path : Path
        The path to the processed data.
    processed_test_set_path : Path
        The path to the processed test set.
    processed_train_set_path : Path
        The path to the processed training set.
    graphs_path : Path
        The path to the graphs.
    graphs_test_set_path : Path
        The path to the graphs for the test set.
    graphs_train_set_path : Path
        The path to the graphs for the training set.
    """

    test_set_str = "test_set"
    train_set_str = "train_set"
    subsets = (test_set_str, train_set_str)
    root: Path
    training_data_file: Path
    stopping_results_file: Path
    processed_path: Path
    processed_test_set_path: Path
    processed_train_set_path: Path
    graphs_path: Path
    graphs_test_set_path: Path
    graphs_train_set_path: Path

    def __init__(self, root: Path):
        """Instantiate the container.

        Parameters
        ----------
        root : Path
            Parent directory for the experimental results, which should already exist.
        """
        self.root = Path(root)

        self.training_data_file = self.root / Path("training_data.csv")
        self.stopping_results_file = self.root / Path("stopping_results.csv")

        self.processed_path = self.root / Path("processed")
        self.processed_test_set_path = self.processed_path / self.test_set_str
        self.processed_train_set_path = self.processed_path / self.train_set_str

        self.graphs_path = self.root / Path("graphs")
        self.graphs_test_set_path = self.graphs_path / self.test_set_str
        self.graphs_train_set_path = self.graphs_path / self.train_set_str

    def __str__(self) -> str:
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        return "\n".join(list(utils.tree(self.root)))

    def setup(self) -> None:
        """Create the output paths for this experiment.

        Raises
        ------
        FileNotFoundError
            If the root does not exist
        """
        if not self.root.exists():
            raise FileNotFoundError(f"The root directory must exist: {self.root.as_posix()}")

        self.processed_path.mkdir(exist_ok=True)
        self.processed_test_set_path.mkdir(exist_ok=True)
        self.processed_train_set_path.mkdir(exist_ok=True)

        self.graphs_path.mkdir(exist_ok=True)
        self.graphs_test_set_path.mkdir(exist_ok=True)
        self.graphs_train_set_path.mkdir(exist_ok=True)

    def teardown(self):
        """Remove everything this object represents."""
        for p in self.root.glob("*"):
            shutil.rmtree(p)


class IndividualOutputDataContainer(OutputDataContainer):
    """Encapsulate the output files from a single AL experiment.

    Attributes
    ----------
    X_unlabeled_pool_file : Path
        The .mtx file containing the features of the unlabeled pool.
    X_test_set_file : Path
        The .mtx file containing the features of the test data.
    y_unlabeled_pool_file : Path
        The .mtx or .npy file containing the labels of the unlabeled pool.
    y_test_set_file : Path
        The .mtx or .npy file containing the labels of the test data.
    model_path : Path
        Path to the trained models.
    batch_path : Path
        Path to the .npy files containing the indices of elements selected at each iteration.
    raw_path : Path
        Path to the raw files produced during evaluation.
    raw_test_set_path : Path
        Path to the raw .json files on the test set produced during evaluation.
    raw_train_set_path : Path
        Path to the raw .json files on the training set produced during evaluation.
    """

    X_unlabeled_pool_file: Path
    X_test_set_file: Path
    y_unlabeled_pool_file: Path
    y_test_set_file: Path
    model_path: Path
    batch_path: Path
    raw_path: Path
    raw_test_set_path: Path
    raw_train_set_path: Path

    def __init__(self, root: Path):
        """Instantiate the container.

        Parameters
        ----------
        root : Path
            Parent directory for the experimental results, which should already exist
        """
        super().__init__(root)

        # Feature vector is a two dimensional dense/sparse matrix, so we use the mtx extension
        self.X_unlabeled_pool_file = self.root / Path("X_unlabeled_pool.mtx")
        self.X_test_set_file = self.root / Path("X_test_set.mtx")

        # Target vector can be one dimensional array or two dimensional dense/sparse matrix
        self.y_test_set_file = self.root / "y_test_set"
        self.y_unlabeled_pool_file = self.root / "y_unlabeled_pool"
        self.set_target_vector_ext()

        self.model_path = self.root / Path("models")
        self.batch_path = self.root / Path("batch")
        self.raw_path = self.root / Path("raw")

        self.raw_test_set_path = self.raw_path / self.test_set_str
        self.raw_train_set_path = self.raw_path / self.train_set_str

    def setup(self) -> None:
        """Create and set up the output paths for this experiment.

        Raises
        ------
        FileNotFoundError
            If the root does not exist
        """
        super().setup()

        self.model_path.mkdir(exist_ok=True)
        self.batch_path.mkdir(exist_ok=True)

        self.raw_path.mkdir(exist_ok=True)
        self.raw_test_set_path.mkdir(exist_ok=True)
        self.raw_train_set_path.mkdir(exist_ok=True)

    def set_target_vector_ext(self, ext: str = None) -> bool:
        """Set the correct file extension for the target vectors.

        Arguments
        ---------
        ext : str
            The extension to use for the target vectors, either ".mtx" or ".npy". If None, will
                glob for existing files and use the extensions of them, if found.

        Returns
        -------
        bool
            True if an extension can be added to the target vector files and False otherwise.
        """
        if ext is not None:
            self.y_test_set_file = self.y_test_set_file.with_suffix(ext)
            self.y_unlabeled_pool_file = self.y_unlabeled_pool_file.with_suffix(ext)
            return True

        matches = list(self.root.glob(f"{self.y_test_set_file.name}.*"))
        if matches:
            self.y_test_set_file = self.y_test_set_file.with_suffix(matches[0].suffix)
            self.y_unlabeled_pool_file = self.y_unlabeled_pool_file.with_suffix(matches[0].suffix)
            return True

        return False


class OutputHelper:
    """Manage the paths for a particular set of AL parameters.

    Attributes
    ----------
    root : Path
        The root directory for the output data.
    dataset_path : Path
        The path to the dataset, which contains the experimental results from individual/avg runs.
    ind_rstates_path : Path
        Path to the individal random state files.
    output_path : Path
        Path to the output of this particular run, accounting for the random state.
    container : IndividualOutputDataContainer
        The container for this particular run, accounting for the random state.
    avg_rstates_path : Path
        Path to the output of the average runs accross all random states.
    avg_container : OutputDataContainer
        The container for the averaged experiments.
    """

    root: Path
    dataset_path: Path
    ind_rstates_path: Path
    output_path: Path
    container: IndividualOutputDataContainer
    avg_rstates_path: Path
    avg_container: OutputDataContainer

    def __init__(self, params: Params) -> None:
        """Manage the paths for a particular set of AL parameters.

        Parameters
        ----------
        params : Params
            Experiment parameters.
        """
        self.root = params.output_root
        self.dataset_path = (
            self.root
            / params.early_stop_mode
            / params.first_batch_mode
            / str(params.batch_size)
            / params.query_strategy
            / params.base_learner
            / params.feature_rep
            / params.dataset
        )

        # Data container for this individual rstates output
        self.ind_rstates_path = self.dataset_path / "ind_rstates"
        self.output_path = self.ind_rstates_path / str(params.random_state)
        self.container = IndividualOutputDataContainer(self.output_path)

        # Data container for the average rstates output
        self.avg_rstates_path = self.dataset_path / "avg_rstates"
        self.avg_container = OutputDataContainer(self.avg_rstates_path)

    def __str__(self) -> str:
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        return "\n".join(list(utils.tree(self.root)))

    def setup(self) -> None:
        """Create, remove, and set up the output paths for this experiment.

        Raises
        ------
        FileNotFoundError
            If the root does not exist
        """
        if not self.root.exists():
            raise FileNotFoundError(f"The root directory must exist: {self.root.as_posix()}")

        self.root.mkdir(exist_ok=True)
        self.dataset_path.mkdir(exist_ok=True, parents=True)

        self.ind_rstates_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
        self.container.setup()

        self.avg_rstates_path.mkdir(exist_ok=True)
        self.avg_container.setup()

    def teardown(self):
        """Remove everything this object represents."""
        for p in self.root.glob("*"):
            shutil.rmtree(p)


class RStatesOutputHelper:
    """Manage the many possible random states for a particular set experiments.

    Attributes
    ----------
    oh : OutputHelper
        An output helper, which contains the paths for the output of this particular run.
    root : Path
        The root directory for the output data, which in this case is nested above the individual
            and average rstate paths.
    avg_rstates_container : OutputDataContainer
        The container for the average rstates output.
    ind_rstates_containers : List[IndividualOutputDataContainer]
        The list of containers for the individual rstates output.
    """

    oh: OutputHelper
    root: Path
    avg_rstates_container: OutputDataContainer
    ind_rstates_containers: List[IndividualOutputDataContainer]

    def __init__(self, oh: OutputHelper, rstates: List[int] = None):
        """Manage the many possible random states for a particular set experiments.

        Parameters
        ----------
        oh : OutputHelper
            OutputHelper object which represent which experiments should be considered.
        rstates : List[int], optional
            An optional subset of rstates to consider, by default None which uses all found rstates.
        """
        self.oh = oh
        self.root = oh.dataset_path
        self.avg_rstates_container = OutputDataContainer(self.oh.avg_rstates_path)

        if rstates is None:
            rstates = list(self.oh.ind_rstates_path.iterdir())
        else:
            rstates = (self.oh.ind_rstates_path / str(i) for i in rstates)

        self.ind_rstates_containers = [OutputDataContainer(p) for p in rstates]

    def __str__(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        return "\n".join(list(utils.tree(self.root)))
