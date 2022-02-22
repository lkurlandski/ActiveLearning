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

def add_stopping_vlines(fig:Figure, ax:Axes, stopping_df:pd.DataFrame) -> Tuple[Figure, Axes]:
    """Add vertical lines to a plot to indicate when stopping methods stopped.

    Parameters
    ----------
    fig : Figure
        matplotlib figure to modifify (learning curve)
    ax : Axes
        matplotlib axes to modifify (learning curve)
    stopping_df : pd.DataFrame
        Stopping results dataframe to extract the performance of stopping methods from

    Returns
    -------
    Tuple[Figure, Axes]
        Modified learning curves
    """

    for i, stopping_method in enumerate(stopping_df):
        ax.vlines(
            x=stopping_df.at["annotations", stopping_method],
            ymin=0,
            ymax=1,
            colors=colors_stopping[i],
            linestyle='dashdot',
            label=stopping_method
        )
    
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

def create_accuracy_graph(accuracy_df:pd.DataFrame) -> Tuple[Figure, Axes]:
    """Create the accuracy vs labeling effort curve.

    Parameters
    ----------
    accuracy_df : pd.DataFrame
        Performance of model per throughout learning

    Returns
    -------
    Tuple[Figure, Axes]
        The learning curve
    """

    return plot_from_dataframe(accuracy_df, x_column="training_data", y_columns=["accuracy"])

def create_weighted_average_graph(weighted_avg_df:pd.DataFrame) -> Tuple[Figure, Axes]:
    """Create the weighted average vs labeling effort curve.

    Parameters
    ----------
    accuracy_df : pd.DataFrame
        Performance of model per throughout learning

    Returns
    -------
    Tuple[Figure, Axes]
        The learning curve
    """

    return plot_from_dataframe(weighted_avg_df, x_column="training_data", y_columns=["f1-score"])

def create_macro_average_graph(macro_avg_df:pd.DataFrame) -> Tuple[Figure, Axes]:
    """Create the macro average vs labeling effort curve.

    Parameters
    ----------
    accuracy_df : pd.DataFrame
        Performance of model per throughout learning

    Returns
    -------
    Tuple[Figure, Axes]
        The learning curve
    """

    return plot_from_dataframe(macro_avg_df, x_column="training_data", y_columns=["f1-score"])

def create_graphs_for_avg(avg_path:Path, stopping_df : pd.DataFrame = None):
    """Create graphs which should exist under the avg directory.

    Parameters
    ----------
    avg_path : Path
        Location of the avg path to create graphs from. Should contain "accuracy", "macro_avg", and
            "weighted_avg" subdirectories
    stopping_df : pd.DataFrame, optional
        Stopping results dataframe to extract the performance of stopping methods from, 
            by default None
    """

    for data_file in ("accuracy", "macro_avg", "weighted_avg"):
        df = pd.read_csv(avg_path / f"{data_file}.csv", index_col=0)

        if data_file == "accuracy":
            fig, ax = create_accuracy_graph(df)
        elif data_file == "macro_avg":
            fig, ax = create_macro_average_graph(df)
        elif data_file == "weighted_avg":
            fig, ax = create_weighted_average_graph(df)

        if stopping_df is not None:
            add_stopping_vlines(fig, ax, stopping_df)

        fig.savefig(avg_path / f"{data_file}.png", dpi=400)
        plt.close()

####################################################################################################

def create_graphs_for_ind(ind_path:Path, stopping_df : pd.DataFrame = None):
    """Create graphs which should exist under the ind directory.

    Parameters
    ----------
    ind_path : Path
        Location of the ind path to create graphs from. Should contain csv files for each category
    stopping_df : pd.DataFrame, optional
        Stopping results dataframe to extract the performance of stopping methods from, 
            by default None
    """

    pass

####################################################################################################

def create_graphs_for_subset(subset_path:Path, stopping_df : pd.DataFrame = None):
    """Create graphs which should exist under each of the train, test, and top_set directories.

    Parameters
    ----------
    subset_path : Path
        Location of the avg path to create graphs from. Should contain "ind", "avg" subdirectories
    stopping_df : pd.DataFrame, optional
        Stopping results dataframe to extract the performance of stopping methods from, 
            by default None
    """

    create_graphs_for_avg(subset_path / "avg", stopping_df)
    create_graphs_for_ind(subset_path / "ind", stopping_df)

####################################################################################################

def create_graphs_for_processed(
        processed_path:Path,
        stopping_file : Path = None,
        stopping_methods : List[str] = ['stabilizing_predictions']
    ) -> None:
    """Create graphs for everything that exists under the processed directory.

    Parameters
    ----------
    processed_path : Path
        Location of the processed path to create graphs from. 
            Should contain "train", "test", "stop_set" subdirectories
    stopping_file : Path
        Location of the stopping results file for applying vertical stopping lines, by default None,
            which will not apply any lines
    stopping_df : pd.DataFrame, optional
        Stopping results dataframe to extract the performance of stopping methods from, 
            by default ['stabilizing_predictions']
    """

    stopping_df = None
    if stopping_file is not None:
        stopping_df = pd.read_csv(stopping_file, index_col=0)
        if stopping_methods is not None:
            stopping_df = stopping_df[stopping_methods]

    for subset in ("test", "train", "stop_set"):
        create_graphs_for_subset(processed_path / subset, stopping_df)

def main(experiment_parameters:Dict[str, Union[str, int]]) -> None:
    """Run the graphing algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    oh = output_helper.OutputHelper(experiment_parameters)
    create_graphs_for_processed(
        oh.ind_rstates_paths['processed_path'],
        oh.ind_rstates_paths['stopping_results_file'],
    )

if __name__ == "__main__":

    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 10,
        "estimator": "svm-ova",
        "dataset": "Avila",
        "random_state": 0,
    }

    main(experiment_parameters)
