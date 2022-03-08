"""Wrapper to run any and all processes for any and all experiment parameters.
"""

import argparse
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import

import driver


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
        "output_root": "./output5",
        "task": "cls",
        "stop_set_size": [1000],
        "batch_size": [7],
        "base_learner": ["SVC"],
        "multiclass": ["ovr"],
        "feature_representation": ["w2v"],
        "dataset": ["20NewsGroups"],
        "random_state": list(range(1)),
    }

    # Controls which part of the program is run
    flags = {"averaging"} if averager else {"active_learning", "processor", "graphing"}

    driver.main(experiment_parameters_lists, flags, local)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--averager", help="Run the averager program instead of pipeline", action="store_true"
    )
    parser.add_argument("--local", help="Run locally, not through SLURM", action="store_true")
    args = parser.parse_args()

    main(args.averager, args.local)
