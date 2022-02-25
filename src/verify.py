"""Searches an experimental configuration for failed or missing files.

# TODO: this script is depreciated
"""

from pathlib import Path
from pprint import pprint
from typing import Dict, Union

import output_helper

def verify_all_runs_successful(experiment_parameters:Dict[str, Union[str, int]]):
    """Ensure that all of the files from an experiment are present in the output directory.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    oh = output_helper.OutputHelper(experiment_parameters)

    first_print_out_performed = False

    files_to_check = [
        #oh.kappa_file,
        #oh.processed_test_accuracy_file,
        #oh.processed_test_macro_avg_file,
        #oh.processed_test_weighted_avg_file,
        #oh.stopping_results_file,
        #oh.avg_processed_accuracy_file,
        #oh.avg_processed_macro_avg_file,
        #oh.avg_processed_weighted_avg_file,
        #oh.avg_stopping_results_file,
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

def main(experiment_parameters:Dict[str, Union[str, int]]):
    """Run the vertification procedure.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    verify_all_runs_successful(experiment_parameters)

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
