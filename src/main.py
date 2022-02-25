"""Main program to execute processed for a specific set of experiment parameters.
"""

import json
from pathlib import Path
from pprint import pformat
from typing import Dict, Set, Union
import warnings

from sklearn.exceptions import ConvergenceWarning

import active_learner
import processor
import averager
import graphing

def main(
        config_file : Path = None,
        experiment_parameters : Dict[str, Union[str, int]]=None,
        flags : Set[str] = None
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
        with open(config_file, 'r') as f:
            experiment_parameters = json.load(f)

    print(f'main.main\nflags:\n{flags},\nexperiment_parameters:\n{pformat(experiment_parameters)}')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if "active_learning" in flags or flags is None:
            active_learner.main(experiment_parameters)
        if "processor" in flags or flags is None:
            processor.main(experiment_parameters)
        if "graphing" in flags or flags is None:
            graphing.main(experiment_parameters)
        # TODO: this should only be run once for all sets of a particular rstate
        if "averaging" in flags or flags is None:
            averager.main(experiment_parameters)

if __name__ == "__main__":

    main(experiment_parameters=
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
