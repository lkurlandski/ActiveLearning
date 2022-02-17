"""Wrapper to run any and all processes for any and all experiment parameters.
"""

import argparse
from pprint import pprint

import driver

# TODO: at the moment, the phase2 flags are run for every possible configuration, which is 
    # completely unnessecary. It only needs to average for one configuration because that will
    # include all configurations.

def main(local):
    
    experiment_parameters_lists = {
        # Only one value permitted
        "output_root": "./output",
        "task": "preprocessedClassification",
        # Iterable of values required
        "stop_set_size": [1000],
        "batch_size": [1],
        "estimator": ["svm"],   # ["mlp", "svm", "svm-ova", "rf"],
        "dataset": ["Iris"],
        "random_state": [0]
    }
    
    # Change the flags to run different parts of the ALL program.
    flags_phase1 = {
        "active_learning",
        "processor",
        "stopping",
    }
    flags_phase2 = {
        "averaging",
        "graphing",
        "verify",
    }
    
    flags = flags_phase1
    
    driver.create_config_files(experiment_parameters_lists, flags, local)
    if not local:
        # TODO: refactor the slurm process to be allow for more elegant naming of jobs
        driver.sbatch_config_files(flags, temp_name=experiment_parameters_lists["dataset"][0])
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Run locally, not through SLURM", action="store_true")
    args = parser.parse_args()
    
    main(args.local)