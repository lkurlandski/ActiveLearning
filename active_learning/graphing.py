"""Create plots for the output data.

TODO
----
- Once the stopping methods module is working properly, add the stopping vlines again.

FIXME
-----
-
"""

from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict, List, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd

from active_learning import output_helper
from active_learning import stopping_methods  # pylint: disable=unused-import

# Colors for stopping methods
colors_stopping = ["lime", "blue", "megenta", "midnightblue"]
# Colors for performance metrics
colors_performance = [
    f"tab:{c}"
    for c in ["blue", "orange", "green", "red", "purple", "brown", "pink", "grey", "olive", "cyan"]
]


def add_stopping_vlines(fig: Figure, ax: Axes, stopping_df: pd.DataFrame) -> Tuple[Figure, Axes]:
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
            linestyle="dashdot",
            label=stopping_method,
        )

    return fig, ax


def plot_from_dataframe(
    df: pd.DataFrame,
    x_column: str = None,
    y_columns: List[str] = None,
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
        raise ValueError(
            f"Too many y_columns to plot and not enough colors:"
            f"\n\t{y_columns}\n\t{colors_performance}"
        )

    x = df.index.to_numpy() if x_column is None else df[x_column]

    fig, ax = plt.subplots()
    for i, col in enumerate(y_columns):
        ax.scatter(x=x, y=df[col], marker=".", color=colors_performance[i], label=col)

    # Assume all performance metrics are 0 to 1
    ax.set_ylim([0, 1])

    ax.legend()
    ax.set_title("Performance vs Labelling Effort")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Performance")

    return fig, ax


####################################################################################################


def create_accuracy_graph(accuracy_df: pd.DataFrame) -> Tuple[Figure, Axes]:
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


def create_weighted_average_graph(weighted_avg_df: pd.DataFrame) -> Tuple[Figure, Axes]:
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


def create_macro_average_graph(macro_avg_df: pd.DataFrame) -> Tuple[Figure, Axes]:
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


def create_graphs_for_overall(overall_path: Path, stopping_df: pd.DataFrame = None):
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
        df = pd.read_csv(overall_path / f"{data_file}.csv", index_col=0)

        if data_file == "accuracy":
            fig, ax = create_accuracy_graph(df)
        elif data_file == "macro_avg":
            fig, ax = create_macro_average_graph(df)
        elif data_file == "weighted_avg":
            fig, ax = create_weighted_average_graph(df)

        if stopping_df is not None:
            add_stopping_vlines(fig, ax, stopping_df)

        fig.savefig(overall_path / f"{data_file}.png", dpi=400)
        plt.close()
        plt.clf()
        plt.cla()
        fig.clf()


####################################################################################################

# TODO: implement
# def create_graphs_for_ind_cat(ind_path:Path, stopping_df : pd.DataFrame = None):
#    pass

####################################################################################################


def create_graphs_for_subset(subset_path: Path, stopping_df: pd.DataFrame = None):
    """Create graphs for the various sets which have been analyzed throughout AL.

    Parameters
    ----------
    subset_path : Path
        The path to the subset analysis folder
    stopping_df : pd.DataFrame, optional
        Stopping results for plotting
    """

    create_graphs_for_overall(
        subset_path / output_helper.OutputDataContainer.overall_str, stopping_df
    )
    # create_graphs_for_ind_cat(subset_path / output_helper.OutputDataContainer.ind_cat_str, stopping_df)


####################################################################################################


def create_graphs_for_container(
    container: output_helper.OutputDataContainer,
    stp_mthd: List[str] = None,
    add_stopping_lines: bool = True,
):
    """Create graphs for a particular data container.

    Parameters
    ----------
    container : output_helper.OutputDataContainer
        The relevant data container
    stp_mthd : List[str], optional
        A list of stopping methods that are column names in stopping_results.csv, for plotting
    """

    if add_stopping_lines:
        stopping_df = pd.read_csv(container.stopping_results_csv_file, index_col=0)
        if stp_mthd is not None:
            stopping_df = stopping_df[stp_mthd]
    else:
        stopping_df = None

    paths = (
        container.processed_stop_set_path,
        container.processed_test_set_path,
        container.processed_train_set_path,
    )
    for path in paths:
        create_graphs_for_subset(path, stopping_df)


def main(experiment_parameters: Dict[str, Union[str, int]]) -> None:
    """Run the graphing algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    print("Beginning Graphing", flush=True)

    oh = output_helper.OutputHelper(experiment_parameters)
    # TODO: Once the stopping methods module is working properly, add the stopping vlines again.
    create_graphs_for_container(
        oh.container,
        None,
        False,  # [repr(stopping_methods.StabilizingPredictions())],
    )

    print("Ending Graphing", flush=True)


if __name__ == "__main__":

    from active_learning import local

    main(local.experiment_parameters)
