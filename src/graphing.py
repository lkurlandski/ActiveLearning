"""Search the output folders for data and create learning curves.
"""

from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
def add_stopping_vlines(path:Path, fig:Figure, ax:Axes, stopping_df:pd.DataFrame):
    
    return fig, ax

def plot_from_dataframe(
        df:pd.DataFrame, 
        x_column:str = None, 
        y_columns:List[str] = None
    ) -> Tuple[Figure, Axes]:
    """Create a plot from a DataFrame of data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of information to plot
    x_column : str, optional
        Indicates which column from the dataframe should be used as the x-axis
    y_columns : List[str], optional
        Indicates which column(s) from the dataframe should be used as the x-axis

    Returns
    -------
    Tuple[Figure, Axes]
        The figure and axes of the plot

    Raises
    ------
    ValueError
        If more y-values are requested than colors are available
    """
    
    y_columns = df.columns.tolist() if y_columns is None else y_columns
    y_columns = y_columns.remove(x_column) if x_column in y_columns else y_columns
    
    if len(y_columns) > len(colors_performance):
        raise ValueError(f"Too many y_columns to plot and not enough colors:"
            f"\n\t{y_columns}\n\t{colors_performance}")

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

def recursively_create_simple_graphs(path:Path) -> None:
    """Search the output structure for csv files and make graphs of them.

    Parameters
    ----------
    path : Path
        Root directory to search
    """
    
    for p in path.iterdir():
        if output_helper.contains_data(p, ignore_raw=True):
            # TODO: incorporate these relative paths into the OutputHelper and eventually replace
                # this code with those paths from the OutputHelper
            stopping_file = p / "stopping" / "results.csv"
            files = [
                p / "processed" / "test" / "avg" / "accuracy",
                p / "processed" / "test" / "avg" / "macro_avg",
                p / "processed" / "test" / "avg" / "weighted_avg"
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

def main(experiment_parameters:Dict[str, Union[str, int]]) -> None:
    """Run the graphing algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """
    
    oh = output_helper.OutputHelper(experiment_parameters)
    recursively_create_simple_graphs(oh.dataset_path)

if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 10,
        "estimator": "mlp",
        "dataset": "Avila",
        "random_state": 0,
    }
    
    main(experiment_parameters)