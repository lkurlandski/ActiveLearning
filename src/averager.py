"""Average the data across multiple sampling versions.
"""

from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

import pandas as pd

import graphing
import output_helper
import stat_helper

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

    # TODO: additionally, average the data across the individual categories
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
            
    graphing.create_graphs_for_processed(oh.avg_rstates_paths['processed_path'])

if __name__ == "__main__":

    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 50,
        "estimator": "svm-ova",
        "dataset": "20NewsGroups",
        "random_state": 0,
    }

    main(experiment_parameters)
