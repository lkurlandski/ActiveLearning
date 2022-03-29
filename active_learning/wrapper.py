"""Wrapper to run any and all processes for any and all experiment parameters.

TODO
----
-

FIXME
-----
-
"""

import argparse
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import

from active_learning import driver
from active_learning import runner


def main(averager: bool, local: bool) -> None:
    """Run the wrapper program to perform experiments in bulk.

    Parameters
    ----------
    averager : bool
        If True, the averager program is executed, instead of the main pipeline
    local : bool
        If True, the experiments should be run locally, not on TCNJ's nodes
    """

    experiment_parameters_lists = {
        "output_root": "/home/hpc/elphicb1/ActiveLearning/ActiveLearning/output",
        "task": "cls",
        "stop_set_size": [1000],
        "batch_size": [200],
        "base_learner": ["SVC"],
        "multiclass": ["ovr"],
        "feature_representation": ["d2v"],
        "dataset": ["20NewsGroups"],
        "random_state": list(range(1)),
    }

    # Controls which part of the program is run
    flags = {"averaging"} if averager else runner.default_flags

    driver.main(experiment_parameters_lists, flags, local)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--averager", help="Run the averager program instead of pipeline", action="store_true"
    )
    parser.add_argument("--local", help="Run locally, not through SLURM", action="store_true")
    args = parser.parse_args()

    main(args.averager, args.local)
