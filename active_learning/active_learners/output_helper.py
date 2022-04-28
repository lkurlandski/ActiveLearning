"""Assist with setting up and maintaining the output structure for experiments.

TODO
----
- Remove the optional arguments from the setup methods.
- Determine a better way of handling the possibility of different file types for the target vector.

FIXME
-----
-
"""

from pathlib import Path
import shutil
from typing import Dict, List, Union

from scipy import sparse
import numpy as np

from active_learning import utils


class OutputDataContainer:
    """Results from one set of experiments or an average across many sets."""

    test_set_str = "test_set"
    train_set_str = "train_set"
    subsets = (test_set_str, train_set_str)

    def __init__(self, root: Path):
        """Instantiate the container.

        Parameters
        ----------
        root : Path
            Parent directory for the experimental results, which should already exist
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
        """Create, remove, and set up the output paths for this experiment.

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
    """Results from a single random state."""

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
        self.X_stop_set_file = self.root / Path("X_stop_set.mtx")

        # Target vector can be one dimensional array or two dimensional dense/sparse matrix
        self.y_test_set_file = self.root / "y_test_set"
        self.y_unlabeled_pool_file = self.root / "y_unlabeled_pool"
        self.y_stop_set_file = self.root / "y_stop_set"
        self.set_target_vector_ext()

        self.model_path = self.root / Path("models")
        self.batch_path = self.root / Path("batch")
        self.raw_path = self.root / Path("raw")

        self.raw_test_set_path = self.raw_path / self.test_set_str
        self.raw_train_set_path = self.raw_path / self.train_set_str

    def setup(self) -> None:
        """Create, remove, and set up the output paths for this experiment.

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
    """Manage the paths for a particular set of AL parameters."""

    def __init__(self, params: Dict[str, Union[str, int, float]]) -> None:
        """Manage the paths for a particular set of AL parameters.

        Parameters
        ----------
        params : Dict[str, Union[str, int, float]]
            Experiment specifications
        """

        params: Dict[str, str] = {k: str(v) for k, v in params.items()}

        self.root = Path(params["output_root"])
        self.dataset_path = (
            self.root
            / params["task"]
            / params["early_stop_mode"]
            / params["first_batch_mode"]
            / params["batch_size"]
            / params["query_strategy"]
            / params["base_learner"]
            / params["feature_representation"]
            / params["dataset"]
        )

        # Data container for this individual rstates output
        self.ind_rstates_path = self.dataset_path / "ind_rstates"
        self.output_path = self.ind_rstates_path / params["random_state"]
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
    """Manage the many possible random states for a particular set experiments."""

    def __init__(self, oh: OutputHelper, rstates: List[int] = None):
        """Manage the many possible random states for a particular set experiments.

        Parameters
        ----------
        oh : OutputHelper
            OutputHelper object which represent which experiments should be considered
        rstates : List[int], optional
            An optional subset of rstates to consider, by default None which uses all found rstates
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
