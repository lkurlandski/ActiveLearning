"""Average the data across multiple sampling versions.
"""

from pathlib import Path

import pandas as pd

import graphing
from output_helper import OutputHelper, contains_data

def mean_dataframes(dfs):
    
    # Concatonates dataframes onto multiple indicies, then groups and averages them
    mean_df = pd.concat(dfs, keys=[i for i in range(len(dfs))]).groupby(level=1).mean()
    
    return mean_df

def average_file_in_directory(path):
    
    # FIXME: replace with output_helper.contains_data(path)
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

def main(experiment_parameters):
    
    output_helper = OutputHelper(**experiment_parameters)
    output_helper.setup_output_path(remove_existing=False, exists_ok=True)
    mean_stopping_df, mean_accuracy_df, mean_macro_avg_df, mean_weighted_avg_df = \
        average_file_in_directory(output_helper.ind_seeds_path)
        
    mean_stopping_df.to_csv(output_helper.avg_stopping_results_file)
    mean_accuracy_df.to_csv(output_helper.avg_processed_accuracy_file)
    mean_macro_avg_df.to_csv(output_helper.avg_processed_macro_avg_file)
    mean_weighted_avg_df.to_csv(output_helper.avg_processed_weighted_avg_file)

if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "/home/hpc/kurlanl1/bloodgood/modAL/output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "initial_pool_size": 10,
        "batch_size": 10,
        "estimator": "mlp",
        "dataset": "Avila",
        "random_seed": 0
    }
    
    main(experiment_parameters)