"""Average the data across multiple sampling versions.
"""

from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

import pandas as pd

import graphing
import output_helper
import stat_helper

# TODO: the averager should only be run once per random state
# TODO: separate functions for averaging across random states, datasets, and other parameters

def main(experiment_parameters:Dict[str, Union[str, int]]) -> None:
    """Run the averaging algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    # Determine which rtates are available for averaging
    oh = output_helper.OutputHelper(experiment_parameters)
    available_rstates = [int(p.name) for p in oh.ind_rstates_path.glob("*")]

    # Average the contents of the processed directory across all the random states
    for subset in ("test", "train", "stop_set"):
        for data_file in ("accuracy", "macro_avg", "weighted_avg"):
            # Attain a list of corresponding files across the rstates
            files_to_avg = []
            for rstate in available_rstates:
                experiment_parameters["random_state"] = rstate
                oh = output_helper.OutputHelper(experiment_parameters)
                files_to_avg.append(oh.ind_rstates_paths[f"processed_{subset}_{data_file}_path"])
            # Average the files, and save them to the average rstate path
            dfs = [pd.read_csv(p, index_col=0) for p in files_to_avg]
            df = stat_helper.mean_dataframes(dfs)
            df.to_csv(oh.avg_rstates_paths[f"processed_{subset}_{data_file}_path"])
            
    # TODO: additionally, average the data across the individual, i.e., raw, categories
    
    # Average the contents of the stoppping directory across all the random states
    files_to_avg = []
    for rstate in available_rstates:
        experiment_parameters["random_state"] = rstate
        oh = output_helper.OutputHelper(experiment_parameters)
        files_to_avg.append(oh.ind_rstates_paths[f"stopping_results_file"])
    dfs = [pd.read_csv(p, index_col=0) for p in files_to_avg]
    df = stat_helper.mean_dataframes(dfs)
    df.to_csv(oh.avg_rstates_paths['stopping_results_file'])
    
    graphing.create_graphs_for_processed(
        oh.avg_rstates_paths['processed_path'],
        oh.avg_rstates_paths['stopping_results_file'],
    )

if __name__ == "__main__":

    experiment_parameters = {
        "output_root": "./output2",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 10,
        "estimator": "mlp",
        "dataset": "Avila",
        "random_state": '13',
    }

    main(experiment_parameters)
