"""Search the output folders for data and create learning curves.

TODO: this is essentially broken...needs significant refactoring.
"""

from operator import index
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
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
        y_columns:List[str] = None,
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

####################################################################################################

def create_accuracy_graph(accuracy_df):

    return plot_from_dataframe(accuracy_df, x_column="training_data", y_columns=["accuracy"])

def create_weighted_average_graph(weighted_average_df):

    return plot_from_dataframe(weighted_average_df, x_column="training_data", y_columns=["f1-score"])

def create_macro_average_graph(macro_average_df):

    return plot_from_dataframe(macro_average_df, x_column="training_data", y_columns=["f1-score"])

def create_graphs_for_avg(avg_path):
    
    for data_file in ("accuracy", "macro_avg", "weighted_avg"):
        df = pd.read_csv(avg_path / f"{data_file}.csv", index_col=0)

        if data_file == "accuracy":
            fig, ax = create_accuracy_graph(df)
        elif data_file == "macro_avg":
            fig, ax = create_macro_average_graph(df)
        elif data_file == "weighted_avg":
            fig, ax = create_weighted_average_graph(df)

        fig.savefig(avg_path / f"{data_file}.png", dpi=400)
        plt.close()

####################################################################################################

def create_graphs_for_ind(ind_path):

    pass

####################################################################################################

def create_graphs_for_subset(subset_path):

    create_graphs_for_avg(subset_path / "avg")
    create_graphs_for_ind(subset_path / "ind")

def create_graphs_for_processed(processed_path):

    for subset in ("test", "train", "stop_set"):
        create_graphs_for_subset(processed_path / subset)

def main(experiment_parameters:Dict[str, Union[str, int]]) -> None:
    """Run the graphing algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    oh = output_helper.OutputHelper(experiment_parameters)
    create_graphs_for_processed(oh.ind_rstates_paths['processed_path'])

if __name__ == "__main__":

    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 50,
        "estimator": "svm-ova",
        "dataset": "20NewsGroups",
        "random_state": 4,
    }

    main(experiment_parameters)
