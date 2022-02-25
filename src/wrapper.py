"""Wrapper to run any and all processes for any and all experiment parameters.
"""

import argparse
from pprint import pprint

import driver

def main(local:bool) -> None:
    """Run the wrapper program to perform experiments in bulk.

    Parameters
    ----------
    local : bool
        If True, the experiments should be run locally, not on TCNJ's nodes.
    """

    experiment_parameters_lists = {
        # Only one value permitted
        "output_root": "./output",
        "task": "preprocessedClassification",
        # Iterable of values required
        "stop_set_size": [1000],
        "batch_size": [7],
        "estimator": ["mlp"],
        "dataset": ["Iris"],
        "random_state": [i for i in range(5)]
    }

    # Change the flags to run different parts of the ALL program.
    flags_phase1 = {
        "active_learning",
        "processor",
        "graphing",
    }
    flags_phase2 = {
        "averaging",
    }

    flags = flags_phase1
    driver.main(experiment_parameters_lists, flags, local)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Run locally, not through SLURM", action="store_true")
    args = parser.parse_args()

    main(args.local)
