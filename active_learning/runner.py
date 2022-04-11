"""Main program to execute processed for a specific set of experiment parameters.

TODO
----
- Decide upon a better naming convention for the flags.

FIXME
-----
-
"""

import argparse
import json
from pathlib import Path
from pprint import pformat
from typing import Dict, Set, Union
import warnings
import os
import signal

from sklearn.exceptions import ConvergenceWarning

from active_learning import active_learner
from active_learning import averager
from active_learning import graphing
from active_learning import local
from active_learning import processor
from active_learning import slurm_handling


all_flags = {"active_learning", "processor", "graphing", "averaging"}
default_flags = {"active_learning", "processor", "graphing"}


def main(
    config_file: Path = None,
    experiment_parameters: Dict[str, Union[str, int]] = None,
    flags: Set[str] = None,
):
    """Run the AL pipeline from a configuration file or from a set of experiment parameters.

    Parameters
    ----------
    config_file : Path, optional
        Location of a configuration file, by default None.
    experiment_parameters : Dict[str, Union[str, int]], optional
        A single set of hyperparmaters and for the active learning experiment, by default None.
    flags : Set[str], optional
        Set of flags to control which parts of the AL pipeline are run, by default None.

    Raises
    ------
    ValueError
        If neither a configuration file or experiment_parameters are given.
    ValueError
        If flags contains an unrecognized flag.
    """

    flags = default_flags if flags is None else flags

    print(f"runner.main\nexperiment_parameters:\n{pformat(experiment_parameters)}")

    if config_file is None and experiment_parameters is None:
        raise ValueError("One of config_file or experiment_parameters must not be None")

    if not flags <= all_flags:
        raise ValueError(
            f"Unexpected flags passed as argument: {flags}." f"\nValid flags are: {all_flags}"
        )

    if experiment_parameters is None:
        with open(config_file, "r", encoding="utf8") as f:
            experiment_parameters = json.load(f)
    
    #writing the files to /local/scratch/
    if "SLURM_JOB_ID" in os.environ:
        #check if the root directory bloodgood exists in /local/scratch/
        #if does not create it, and make output_root /local/scratch/bloodgood/
        #when the job finsihes write the files to the user path using glob
        if not(os.path.isdir(slurm_handling.node_root)):
            os.mkdir(slurm_handling.node_root)
        slurm_handling.user_path = experiment_parameters["output_root"]
        experiment_parameters["output_root"] = slurm_handling.node_root

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if "active_learning" in flags:
            active_learner.main(experiment_parameters)
            slurm_handling.move_output() 
            experiment_parameters["output_root"] = slurm_handling.user_path

    if "processor" in flags:
        processor.main(experiment_parameters)

    if "graphing" in flags:
        graphing.main(experiment_parameters)

    if "averaging" in flags:
        averager.main(experiment_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--averager", help="Run the averager program instead of pipeline", action="store_true"
    )
    args = parser.parse_args()

    main(
        config_file=None,
        experiment_parameters=local.experiment_parameters,
        flags=default_flags if not args.averager else "averaging",
    )
