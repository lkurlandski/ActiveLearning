"""Assist with setting up and maintaining the output structure for experiments.
"""

from pathlib import Path
import shutil
import os
from typing import Dict, List, Union

from utils import DisplayablePath

# TODO: extract the concept of model evaluation upon a single set of examples into a container class
# TODO: modify some of the function is graphing to operate upon these containers
# TODO: separate the average OutputDataContainer (has no raw) from the indivudal OutputDataContainer
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

        ret = []
        paths = DisplayablePath.make_tree(self.root)
        for path in paths:
            ret.append(path.displayable())

        return "\n".join(ret)

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

    def __init__(self, experiment_parameters: Dict[str, Union[str, int]]):
        """Manage the paths for a particular set of AL parameters.

        Parameters
        ----------
        experiment_parameters : Dict[str, Union[str, int]]
            Experiment specifications
        """

        # Shorthand for the experiment parameters
        self.experiment_parameters = experiment_parameters

        #self.user_path = Path(experiment_parameters["output_root"])

        # Base paths
        self.root = Path(experiment_parameters["output_root"])
        self.slurm_path = self.root / os.environ['SLURM_JOB_ID']
        self.task_path = self.slurm_path / experiment_parameters["task"]
        self.stop_set_size_path = self.task_path / str(experiment_parameters["stop_set_size"])
        self.batch_size_path = self.stop_set_size_path / str(experiment_parameters["batch_size"])
        self.estimator_path = self.batch_size_path / experiment_parameters["estimator"]
        self.dataset_path = self.estimator_path / experiment_parameters["dataset"]

        # Data container for this individual rstates output
        self.ind_rstates_path = self.dataset_path / "ind_rstates"
        self.output_path = self.ind_rstates_path / str(experiment_parameters["random_state"])
        self.container = OutputDataContainer(self.output_path)

        # Data container for the average rstates output
        self.avg_rstates_path = self.dataset_path / "avg_rstates"
        self.avg_container = OutputDataContainer(self.avg_rstates_path)

    def __str__(self) -> str:
        """Return a unix tree-like representation of this instance.

        Returns
        -------
        str
            str representation
        """

        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        ret = []
        paths = DisplayablePath.make_tree(self.root)
        for path in paths:
            ret.append(path.displayable())

        return "\n".join(ret)

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
        self.slurm_path.mkdir(exist_ok=exist_ok)
        self.task_path.mkdir(exist_ok=exist_ok)
        self.stop_set_size_path.mkdir(exist_ok=exist_ok)
        self.batch_size_path.mkdir(exist_ok=exist_ok)
        self.estimator_path.mkdir(exist_ok=exist_ok)
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

    def move_output_(self, user_path, job_id_list):
        #print(user_path)
        #local_node_path = "/local/scratch/output"
        #shutil.move(local_node_path, user_path)

        source_dir = "/local/scratch/"
    
        for job_id in job_id_list:
            shutil.move(source_dir + job_id, user_path)
        
        self.experiment_parameters["output_root"] = user_path


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
        """Return a unix tree-like representation of this instance.

        Returns
        -------
        str
            str representation
        """

        if not self.root.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root.as_posix()}")

        ret = []
        paths = DisplayablePath.make_tree(self.root)
        for path in paths:
            ret.append(path.displayable())

        return "\n".join(ret)


def test():
    """Test."""

    odc = OutputDataContainer("./tmp")
    odc.setup()
    print(odc)
    odc.teardown()

    oh = OutputHelper(
        experiment_parameters={
            "output_root": "./tmp",
            "task": "cls",
            "stop_set_size": 1000,
            "batch_size": 10,
            "estimator": "mlp",
            "dataset": "Iris",
            "random_state": 1,
        }
    )
    oh.setup()
    print(oh)
    oh.teardown()

    oh1 = OutputHelper(
        experiment_parameters={
            "output_root": "./tmp",
            "task": "cls",
            "stop_set_size": 1000,
            "batch_size": 10,
            "estimator": "mlp",
            "dataset": "Iris",
            "random_state": 0,
        }
    )
    oh1.setup()
    oh2 = OutputHelper(
        experiment_parameters={
            "output_root": "./tmp",
            "task": "cls",
            "stop_set_size": 1000,
            "batch_size": 10,
            "estimator": "mlp",
            "dataset": "Iris",
            "random_state": 1,
        }
    )
    oh2.setup()
    orsc = RStatesOutputHelper(oh.dataset_path)
    print(orsc)
    oh1.teardown()
    oh2.teardown()

def move_output(user_path):
    #print(user_path)
    #local_node_path = "/local/scratch/output"
    #shutil.move(local_node_path, user_path)

    source_dir = "/local/scratch/output"
    target_dir = user_path
    
    file_names = os.listdir(source_dir)
    
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)


if __name__ == "__main__":
    test()
