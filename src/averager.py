"""Average the data across multiple sampling versions.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

import graphing
import output_helper

def mean_dataframes(dfs:List[pd.DataFrame]) -> pd.DataFrame:
    """Return the ``mean'' of a dataframe, taken in an element-wise fashion.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        Dataframes to operate upon.

    Returns
    -------
    pd.DataFrame
        Meaned dataframe.
    """
    
    # Concatonates dataframes onto multiple indicies, then groups and averages them
    mean_df = pd.concat(dfs, keys=[i for i in range(len(dfs))]).groupby(level=1).mean()
    
    return mean_df

def average_file_in_directory(
        path:Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average csv files with other csv files at the same level.

    Parameters
    ----------
    path : Path
        Head directory to perform the recursive search from.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The average of the stopping, accuracy, macro_avg, and weighted_avg csv files.
    """
    
    # TODO: replace with output_helper.contains_data(path)
    # Base case: finds a raw, processed, and stopping directories
    files = set((p.name for p in path.glob("*")))
    
    if "raw" in files and "processed" in files and "stopping" in files:
        stopping_df = pd.read_csv(path / "stopping" / "results.csv")
        accuracy_df = pd.read_csv(path / "processed" / "average" / "accuracy.csv")
        macro_avg_df = pd.read_csv(path / "processed" / "average" / "macro_avg.csv")
        weighted_avg_df = pd.read_csv(path / "processed" / "average" / "weighted_avg.csv")
        
        return stopping_df, accuracy_df, macro_avg_df, weighted_avg_df
    
    stopping_dfs, accuracy_dfs, macro_avg_dfs, weighted_avg_dfs = [], [], [], []
    
    for f in sorted(p for p in path.glob("*")):
        if f.is_dir():
            try:
                stopping_df, accuracy_df, macro_avg_df, weighted_avg_df = average_file_in_directory(f)
            except FileNotFoundError as e:
                print(f"Caught and ignoring error:\n{e}")
                continue
            
            stopping_dfs.append(stopping_df)
            accuracy_dfs.append(accuracy_df)
            macro_avg_dfs.append(macro_avg_df)
            weighted_avg_dfs.append(weighted_avg_df)
        
    mean_stopping_df = mean_dataframes(stopping_dfs)
    mean_accuracy_df = mean_dataframes(accuracy_dfs)
    mean_macro_avg_df = mean_dataframes(macro_avg_dfs)
    mean_weighted_avg_df = mean_dataframes(weighted_avg_dfs)
    
    return mean_stopping_df, mean_accuracy_df, mean_macro_avg_df, mean_weighted_avg_df

def main(experiment_parameters:Dict[str, Union[str, int]]) -> None:
    """Run the averaging algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """
    
    oh = output_helper.OutputHelper(experiment_parameters)
    oh.setup_output_path(remove_existing=False, exist_ok=True)
    mean_stopping_df, mean_accuracy_df, mean_macro_avg_df, mean_weighted_avg_df = \
        average_file_in_directory(oh.ind_rstates_path)
        
    mean_stopping_df.to_csv(oh.avg_stopping_results_file)
    mean_accuracy_df.to_csv(oh.avg_processed_accuracy_file)
    mean_macro_avg_df.to_csv(oh.avg_processed_macro_avg_file)
    mean_weighted_avg_df.to_csv(oh.avg_processed_weighted_avg_file)

if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 10,
        "estimator": "svm-ova",
        "dataset": "Iris",
        "random_state": 0,
    }
    
    main(experiment_parameters)