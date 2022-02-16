"""Search the output folders for data and create learning curves.
"""

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import output_helper

# Colors for stopping methods
colors_stopping = ['lime', 'blue', 'megenta', 'midnightblue']
# Colors for performance metrics
colors_performance = [
    f'tab:{c}' for c in 
    ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan']
]

# TODO: implement this function
def add_stopping_vlines(path, fig, ax, stopping_df):
    
    if stopping_df is None:
        return fig, ax
    
    return fig, ax

def plot_from_dataframe(df, x_column=None, y_columns=None):
    
    y_columns = df.columns.tolist() if y_columns is None else y_columns
    y_columns = y_columns.remove(x_column) if x_column in y_columns else y_columns
    
    if len(y_columns) > len(colors_performance):
        raise ValueError(f"Too many y_columns to plot and not enough colors:\n\t{y_columns}\n\t{colors_performance}")

    x = df.index.to_numpy() if x_column is None else df[x_column]

    fig, ax = plt.subplots()
    for i, col in enumerate(y_columns):
        ax.scatter(x=x, y=df[col], marker='.', color=colors_performance[i], label=col)
    
    # Assume all performance metrics are 0 to 1
    ax.set_ylim([0, 1])
    
    ax.legend()
    ax.set_title("Performance vs Labelling Effort")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Performance")
    
    return fig, ax

def recursively_create_simple_graphs(path):
    
    for p in path.iterdir():
        if output_helper.contains_data(p, ignore_raw=True):
            # TODO: incorporate these relative paths into the OutputHelper and eventually replace
                # this code with those paths from the OutputHelper
            stopping_file = p / "stopping" / "results.csv"
            files = [
                p / "processed" / "average" / "accuracy",
                p / "processed" / "average" / "macro_avg",
                p / "processed" / "average" / "weighted_avg"
            ]
            
            # TODO: remove the checks for files existing and use verify.verify_all_runs_successful
            for f in files + [stopping_file]:
                if not f.with_suffix('.csv').exists():
                    print(f"Caught and ignoring error:\n{FileNotFoundError(f.as_posix())}")
                    
            stopping_df = pd.read_csv(stopping_file) if stopping_file.exists() else None
            
            for f in files:                    
                df = pd.read_csv(f.with_suffix('.csv'))
                # TODO: indicate the x_column should be 'labels'
                fig, ax = plot_from_dataframe(df, x_column=None)
                fig, ax = add_stopping_vlines(p, fig, ax, stopping_df)
                fig.savefig(f.with_suffix('.png'), dpi=400)
                plt.close()
        else:
            recursively_create_simple_graphs(p)

def main(experiment_parameters):
    
    oh = output_helper.OutputHelper(experiment_parameters)
    recursively_create_simple_graphs(oh.dataset_path)

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