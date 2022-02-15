"""Wrapper to run any and all processes for any and all experiment parameters.

Usage:
    > python src/wrapper.py
    > python src/wrapper.py --local
"""

import argparse
from pprint import pprint

import driver

def main(local):
    
    experiment_parameters_lists = {
        # Only one value permitted
        "output_root": "./src/output",
        "task": "preprocessedClassification",
        # Iterable of values required
        "stop_set_size": [1000],
        "initial_pool_size": [10],
        "batch_size": [100],
        "estimator": ["svm-ova"],   # ["mlp", "svm", "svm-ova", "rf"],
        "dataset": ["20NewsGroups-raw"],
        "random_seed": [0, 1, 2, 3, 4]
    }
    
    # Change the flags to run different parts of the ALL program.
    flags = {
        #"active_learning", 
        #"analysis", 
        #"stopping", 
        #"averaging",
        "verify",
    }
    
    driver.create_config_files(experiment_parameters_lists, flags, local)
    if not local:
        driver.sbatch_config_files(flags)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="Run locally, not through SLURM", action="store_true")
    args = parser.parse_args()
    
    main(args.local)