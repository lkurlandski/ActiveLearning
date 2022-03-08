"""Main program to execute processed for a specific set of experiment parameters.
"""

import json
from pathlib import Path
from pprint import pformat
from typing import Dict, Set, Union
import warnings
import os

from sklearn.exceptions import ConvergenceWarning

import active_learner
import config
import averager
import graphing
import processor
import utils
import output


def main(
    config_file: Path = None,
    experiment_parameters: Dict[str, Union[str, int]] = None,
    flags: Set[str] = None,
):
    """Run the AL pipeline from a configuration file or from a set of experiment parameters.

    Parameters
    ----------
    config_file : Path, optional
        Location of a configuration file, by default None
    experiment_parameters : Dict[str, Union[str, int]], optional
        A single set of hyperparmaters and for the active learning experiment, by default None
    flags : Set[str], optional
        Set of flags to control which parts of the AL pipeline are run, by default None

    Raises
    ------
    ValueError
        If neither a configuration file or experiment_parameters are given
    """

    if config_file is None and experiment_parameters is None:
        raise ValueError("One of config_file or experiment_parameters must not be None")

    if experiment_parameters is None:
        with open(config_file, "r", encoding="utf8") as f:
            experiment_parameters = json.load(f)

    print(
        f"runner.main\nflags:\n{flags},\nexperiment_parameters:\n{pformat(experiment_parameters)}"
    )

    utils.print_memory_stats(True)

    job_id_list = []
    print(os.environ["SLURM_JOB_ID"])
    try:
        job_id_list.append(os.environ["SLURM_JOB_ID"])
        user_path = experiment_parameters["output_root"]
        experiment_parameters["output_root"] = config.node_path
    except Exception:
        #print("Running locally.")
        pass

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if "active_learning" in flags or flags is None:
            active_learner.main(experiment_parameters)
            output.move_output(experiment_parameters, user_path, job_id_list)
        if "processor" in flags or flags is None:
            processor.main(experiment_parameters)
        if "graphing" in flags or flags is None:
            graphing.main(experiment_parameters)
        # TODO: this should only be run once for all sets of a particular rstate
        if "averaging" in flags or flags is None:
            averager.main(experiment_parameters)


if __name__ == "__main__":

    main(config.experiment_parameters)
