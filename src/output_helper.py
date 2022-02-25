"""Assist with setting up and maintaining the output structure for experiments.

TODO: significant refactoring...suggest to reduce the deep nesting and use a single processed csv.
"""

import itertools
from pathlib import Path
import shutil
from typing import Dict, Union

from utils import DisplayablePath

# TODO: currently, this runs the experiments on the hpc nodes, but writes files locally.
    # Develop a system (along with the output helper) that writes the experiment
    # to the /local/scratch directory on the node, then copies them over to the local directory
    # when complete.

# TODO: replace report_train_path etc. with report_unlabeled_pool_path

def contains_data(path:Path, ignore_raw : bool = False) -> bool:
    """Determine if a particular directory contains experiment data or not.

    FIXME: this no longer works

    Parameters
    ----------
    path : Path
        Directory to check
    ignore_raw : bool, optional
        Ignore the directory named raw, by default False

    Returns
    -------
    bool
        If the directory contains data or not
    """

    files = set((p.name for p in path.glob("*")))

    if "processed" in files and "stopping" in files:
        if ignore_raw:
            return True
        if "raw" in files:
            return True

    return False

def get_structure_beneath_raw(raw_path:Path) -> Dict[str, Path]:
    """Return the dictionary of path names that should occur beneath the raw directory.

    Parameters
    ----------
    raw_path : Path
        Location of the raw path in question

    Returns
    -------
    Dict[str, Path]
        Named structure beneath raw dictionary
    """

    d = {
        'kappa_file' : raw_path / "kappas.csv",
        'num_training_data_file' : raw_path / "num_training_data.csv",
    }

    for subset in ("test", "train", "stop_set"):
        d[f"report_{subset}_path"] = raw_path / subset

    return d

def get_structure_beneath_processed(processed_path:Path) -> Dict[str, Path]:
    """Return the dictionary of path names that should occur beneath the processed directory.

    Parameters
    ----------
    processed_path : Path
        Location of the processed path in question

    Returns
    -------
    Dict[str, Path]
        Named structure beneath processed dictionary
    """

    d = {}

    for subset in ("test", "train", "stop_set"):
        d[f"processed_{subset}_path"] = processed_path / subset
        d[f"processed_{subset}_ind_path"] = processed_path / subset / "ind"
        d[f"processed_{subset}_avg_path"] = processed_path / subset / "avg"
        for data_file in ("accuracy", "macro_avg", "weighted_avg"):
            d[f"processed_{subset}_{data_file}_path"] = \
                d[f"processed_{subset}_avg_path"] / f"{data_file}.csv"# TODO: should be file, not path

    return d

def get_structure_beneath_stopping(stopping_path:Path) -> Dict[str, Path]:
    """Return the dictionary of path names that should occur beneath the stopping directory.

    Parameters
    ----------
    stopping_path : Path
        Location of the stopping path in question

    Returns
    -------
    Dict[str, Path]
        Named structure beneath stopping dictionary
    """

    d = {'stopping_results_file' : stopping_path / "results.csv"}

    return d

def get_ind_avg_rstates_structure(parent:Path) -> Dict[str, Path]:
    """Return the dictionary of path names that should occur beneath the ind/avg rstates directory.

    Parameters
    ----------
    parent : Path
        Parent of the ind/avg rstates in question

    Returns
    -------
    Dict[str, Path]
        Named structure beneath ind/avg rstates dictionary
    """

    d = {
        'raw_path' : parent / "raw",
        'processed_path' : parent / "processed",
        'stopping_path' : parent / "stopping",
    }

    d.update(get_structure_beneath_raw(d['raw_path']))
    d.update(get_structure_beneath_processed(d['processed_path']))
    d.update(get_structure_beneath_stopping(d['stopping_path']))

    return d

class OutputHelper:

    # TODO: create functions for specific tasks that modify the path in between the base path and
        # the portion where data is stored.

    required_experiment_parameters = {
        "output_root",
        "task",
        "stop_set_size",
        "batch_size",
        "estimator",
        "dataset",
        "random_state"
    }

    def __init__(self, experiment_parameters:Dict[str, Union[str, int]]):
        """Helper to manage the output structure and paths of experiments.

        Parameters
        ----------
        experiment_parameters : Dict[str, Union[str, int]], optional
            A single set of hyperparmaters and for the active learning experiment

        Raises
        ------
        ValueError
            If not all of required_experiment_parameters are contained in experiment_parameters
        """

        # TODO: move all checking behavior into its own file.
        # Ensure every required experiment parameter is present
        if set(experiment_parameters.keys()) != self.required_experiment_parameters:
            raise ValueError(f"Expected the following parameters:\n\t"
                f"{sorted(self.required_experiment_parameters)}\nbut recieved the following:\n\t"
                f"{sorted(experiment_parameters.keys())}"
            )

        # Shorthand for the experiment parameters
        self.experiment_parameters = experiment_parameters
        e = self.experiment_parameters

        # Base part of the path
        self.output_root = Path(e['output_root'])
        self.task_path = self.output_root / e['task']
        self.stop_set_size_path = self.task_path / str(e['stop_set_size'])
        self.batch_size_path = self.stop_set_size_path / str(e['batch_size'])
        self.estimator_path = self.batch_size_path / e['estimator']
        self.dataset_path = self.estimator_path / e['dataset']

        # Individual and average rstate paths
        self.ind_rstates_path = self.dataset_path / "ind_rstates"
        self.avg_rstates_path = self.dataset_path / "avg_rstates"

        # Output path of a particular experiment lands here
        self.output_path = self.ind_rstates_path / str(e['random_state'])

        # Defines a directories under these two major branches
        self.ind_rstates_paths = get_ind_avg_rstates_structure(self.output_path)
        self.avg_rstates_paths = get_ind_avg_rstates_structure(self.avg_rstates_path)

    def __str__(self) -> str:
        """Return a unix tree-like representation of this instance.

        Returns
        -------
        str
            str representation
        """

        ret = []
        paths = DisplayablePath.make_tree(self.dataset_path)
        for path in paths:
            ret.append(path.displayable())

        return "\n".join(ret)

    def setup_output_path(self, remove_existing : bool = False, exist_ok : bool = True) -> None:
        """Create, remove, and set up the output paths for this experiment.

        Parameters
        ----------
        remove_existing : bool, optional
            If True, removes the existing structure, by default False
        exist_ok : bool, optional
            If True, does not raise errors if directory exists, by default True

        Raises
        ------
        FileExistsError
            If the paths exists, but remove_existing not True
        """

        if self.output_path.exists():
            if remove_existing:
                shutil.rmtree(self.output_path)
            elif exist_ok:
                pass
            else:
                raise FileExistsError(f"{self.output_path} exists.")

        self.dataset_path.mkdir(parents=True, exist_ok=exist_ok)
        self.ind_rstates_path.mkdir(exist_ok=exist_ok)
        self.avg_rstates_path.mkdir(exist_ok=exist_ok)
        self.output_path.mkdir(exist_ok=exist_ok)

        for path in self.ind_rstates_paths.values():
            if path.suffix == '':
                path.mkdir(exist_ok=exist_ok)

        for path in self.avg_rstates_paths.values():
            if path.suffix == '':
                path.mkdir(exist_ok=exist_ok)

if __name__ == "__main__":

    oh = OutputHelper(experiment_parameters=
        {
            "output_root": "./output",
            "task": "cls",
            "stop_set_size": 1000,
            "batch_size": 10,
            "estimator": "mlp",
            "dataset": "Avila",
            "random_state": 0,
        }
    )
    oh.setup_output_path()

    print(str(oh))
