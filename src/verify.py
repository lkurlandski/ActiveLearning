"""Searches an experimental configuration for failed or missing files.
"""

from pathlib import Path
from pprint import pprint

import output_helper

def verify_all_runs_successful(experiment_parameters):
    
    oh = output_helper.OutputHelper(experiment_parameters)

    first_print_out_performed = False
    
    files_to_check = [
        oh.kappa_file,
        oh.processed_accuracy_file,
        oh.processed_macro_avg_file,
        oh.processed_weighted_avg_file,
        oh.stopping_results_file,
        oh.avg_processed_accuracy_file,
        oh.avg_processed_macro_avg_file,
        oh.avg_processed_weighted_avg_file,
        oh.avg_stopping_results_file,
    ]
    for f in files_to_check:
        if not f.exists():
            if not first_print_out_performed:
                print("Verifying the following experiment parameters:")
                pprint(experiment_parameters)
                print("Found the following files to be missing:")
                first_print_out_performed = True

            print(f"\t{f}")
    print()
    
def main(experiment_parameters):
    
    verify_all_runs_successful(experiment_parameters)
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "/home/hpc/kurlanl1/bloodgood/modAL/output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "initial_pool_size": 10,
        "batch_size": 10,
        "estimator": "mlp",
        "dataset": "Avila",
        "random_state": 0
    }
    
    main(experiment_parameters)