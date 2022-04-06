"""Assist with setting up and maintaining the output structure for experiments.

TODO
----
- This system will probably require some refactoring at some point.

FIXME
-----
-
"""

from copy import deepcopy
from pathlib import Path
import shutil
import os
from typing import Dict, List, Union

#import utils
#import config
from active_learning import utils


class OutputDataContainer:
    """Results from one set of experiments or an everage across many sets."""

    stop_set_str = "stop_set"
    test_set_str = "test_set"
    train_set_str = "train_set"
    subsets = (stop_set_str, test_set_str, train_set_str)

    ind_cat_str = "ind"
    overall_str = "overall"

    def __init__(self, root: Path):
        """Results from one set of experiments or an everage across many sets.

        Parameters
        ----------
        root : Path
            Parent directory for the experimental results, which should already exist
        """

        self.root = Path(root)

        self.training_data_file = self.root / Path("training_data.csv")
        self.stopping_results_json_file = self.root / Path("stopping_results.json")
        self.stopping_results_csv_file = self.root / Path("stopping_results.csv")

        self.processed_path = self.root / Path("processed")
        self.raw_path = self.root / Path("raw")
        print("This is root", self.root)

        self.raw_stop_set_path = self.raw_path / self.stop_set_str
        self.raw_test_set_path = self.raw_path / self.test_set_str
        self.raw_train_set_path = self.raw_path / self.train_set_str

        self.processed_stop_set_path = self.processed_path / self.stop_set_str
        self.processed_test_set_path = self.processed_path / self.test_set_str
        self.processed_train_set_path = self.processed_path / self.train_set_str

        self.processed_stop_set_ind = self.processed_stop_set_path / self.ind_cat_str
        self.processed_stop_set_all = self.processed_stop_set_path / self.overall_str

        self.processed_test_set_ind = self.processed_test_set_path / self.ind_cat_str
        self.processed_test_set_all = self.processed_test_set_path / self.overall_str

        self.processed_train_set_ind = self.processed_train_set_path / self.ind_cat_str
        self.processed_train_set_all = self.processed_train_set_path / self.overall_str

    def __str__(self) -> str:
        """Return a unix tree-like representation of this instance.

        Returns
        -------
        str
            str representation
        """

        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        return "\n".join(list(utils.tree(self.root)))

    def setup(self, remove_existing: bool = False, exist_ok: bool = True) -> None:
        """Create, remove, and set up the output paths for this experiment.

        Parameters
        ----------
        remove_existing : bool, optional
            If True, removes the existing structure, by default False
        exist_ok : bool, optional
            If True, does not raise errors if directory exists, by default True

        Raises
        ------
        FileNotFoundError
            If the root does not exist
        """

        if not self.root.exists():
            raise FileNotFoundError(f"The root directory must exist: {self.root.as_posix()}")

        if remove_existing:
            shutil.rmtree(self.root)
            self.root.mkdir()

        self.processed_path.mkdir(exist_ok=exist_ok)
        self.raw_path.mkdir(exist_ok=exist_ok)

        self.raw_stop_set_path.mkdir(exist_ok=exist_ok)
        self.raw_test_set_path.mkdir(exist_ok=exist_ok)
        self.raw_train_set_path.mkdir(exist_ok=exist_ok)

        self.processed_stop_set_path.mkdir(exist_ok=exist_ok)
        self.processed_test_set_path.mkdir(exist_ok=exist_ok)
        self.processed_train_set_path.mkdir(exist_ok=exist_ok)

        self.processed_stop_set_ind.mkdir(exist_ok=exist_ok)
        self.processed_stop_set_all.mkdir(exist_ok=exist_ok)

        self.processed_test_set_ind.mkdir(exist_ok=exist_ok)
        self.processed_test_set_all.mkdir(exist_ok=exist_ok)

        self.processed_train_set_ind.mkdir(exist_ok=exist_ok)
        self.processed_train_set_all.mkdir(exist_ok=exist_ok)

    def teardown(self):
        """Remove everything this object represents."""

        for p in self.root.glob("*"):
            shutil.rmtree(p)


class OutputHelper:
    """Manage the paths for a particular set of AL parameters."""

    def __init__(self, experiment_parameters: Dict[str, Union[str, int, float]]) -> None:
        """Manage the paths for a particular set of AL parameters.

        Parameters
        ----------
        experiment_parameters : Dict[str, Union[str, int, float]]
            Experiment specifications
        """

        # TODO: can this be safely removed? Is it being used anywhere?
        self.experiment_parameters = experiment_parameters

        # Path objects require string parameters for everything
        experiment_parameters: Dict[str, str] = {
            k: str(v) for k, v in deepcopy(experiment_parameters).items()
        }

        # Base paths
        self.root = Path(experiment_parameters["output_root"])
        print(self.root)
        self.task_path = self.root / experiment_parameters["task"]
        self.stop_set_size_path = self.task_path / experiment_parameters["stop_set_size"]
        self.batch_size_path = self.stop_set_size_path / experiment_parameters["batch_size"]
        self.query_strategy_path = self.batch_size_path / experiment_parameters["query_strategy"]
        self.base_learner_path = self.query_strategy_path / experiment_parameters["base_learner"]
        self.multiclass_path = self.base_learner_path / experiment_parameters["multiclass"]
        self.feature_representation_path = (
            self.multiclass_path / experiment_parameters["feature_representation"]
        )
        self.dataset_path = self.feature_representation_path / experiment_parameters["dataset"]

        # Data container for this individual rstates output
        self.ind_rstates_path = self.dataset_path / "ind_rstates"
        self.output_path = self.ind_rstates_path / experiment_parameters["random_state"]
        self.container = OutputDataContainer(self.output_path)

        # Data container for the average rstates output
        self.avg_rstates_path = self.dataset_path / "avg_rstates"
        self.avg_container = OutputDataContainer(self.avg_rstates_path)

    def __str__(self) -> str:

        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        return "\n".join(list(utils.tree(self.root)))

    def setup(self, remove_existing: bool = False, exist_ok: bool = True) -> None:
        """Create, remove, and set up the output paths for this experiment.

        Parameters
        ----------
        remove_existing : bool, optional
            If True, removes the existing structure, by default False
        exist_ok : bool, optional
            If True, does not raise errors if directory exists, by default True

        Raises
        ------
        FileNotFoundError
            If the root does not exist
        """

        if not self.root.exists():
            raise FileNotFoundError(f"The root directory must exist: {self.root.as_posix()}")

        if remove_existing:
            shutil.rmtree(self.root)
            self.root.mkdir()

        self.root.mkdir(exist_ok=exist_ok)
        #self.slurm_path.mkdir(exist_ok=exist_ok)
        self.task_path.mkdir(exist_ok=exist_ok)
        self.stop_set_size_path.mkdir(exist_ok=exist_ok)
        self.batch_size_path.mkdir(exist_ok=exist_ok)
        self.query_strategy_path.mkdir(exist_ok=exist_ok)
        self.base_learner_path.mkdir(exist_ok=exist_ok)
        self.multiclass_path.mkdir(exist_ok=exist_ok)
        self.feature_representation_path.mkdir(exist_ok=exist_ok)
        self.dataset_path.mkdir(exist_ok=exist_ok)

        self.ind_rstates_path.mkdir(exist_ok=exist_ok)
        self.output_path.mkdir(exist_ok=exist_ok)
        self.container.setup(remove_existing, exist_ok)

        self.avg_rstates_path.mkdir(exist_ok=exist_ok)
        self.avg_container.setup(remove_existing, exist_ok)

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
