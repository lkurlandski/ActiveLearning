"""Main program to execute processed for a specific set of experiment parameters.
"""

import argparse
import json
import os
from pathlib import Path
from pprint import pformat
import shutil
from typing import Dict, Set, Union
import warnings

from sklearn.exceptions import ConvergenceWarning

import active_learner
import config
import averager
import graphing
import processor
import utils


def main(
    *,
    config_file: Path = None,
    experiment_parameters: Dict[str, Union[str, int]] = None,
    flags: Set[str] = None,
    hpc: bool = True,
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
    hpc : bool, optional
        If True, assumes this script is being run through a SLURM job scheduler

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

    flags = {"active_learning", "processor", "graphing"} if flags is None else flags

    print(
        f"runner.main\n-----------\n"
        f"flags:{flags},\n"
        f"experiment_parameters:\n{pformat(experiment_parameters)}"
    )

    utils.print_memory_stats(True)

    if "active_learning" in flags:
        d = {}
        if hpc:
            tmp_root = Path(f"/local/scratch/{os.environ['SLURM_JOB_ID']}")
            tmp_root.mkdir()
            d["output_root"] = tmp_root.as_posix()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            active_learner.main({**experiment_parameters, **d})
        shutil.move(d["output_root"], experiment_parameters["output_root"])
    if "processor" in flags:
        processor.main(experiment_parameters)
    if "graphing" in flags:
        graphing.main(experiment_parameters)
    if "averaging" in flags:
        averager.main(experiment_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Run locally, not through SLURM", action="store_true")
    args = parser.parse_args()

    main(experiment_parameters=config.experiment_parameters, hpc=(not args.local))
