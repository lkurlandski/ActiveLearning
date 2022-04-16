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

from active_learning.active_learners import output_helper
#from active_learning import stopping_methods  # pylint: disable=unused-import

# Colors for stopping methods
colors_stopping = ["lime", "blue", "megenta", "midnightblue"]
# Colors for performance metrics
colors_performance = [
    f"tab:{c}"
    for c in [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "grey",
        "olive",
        "cyan",
    ]
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

    ax.set_ylim([0, 1])

    ax.legend()
    ax.set_title("Performance vs Labelling Effort")
    ax.set_xlabel("Labels")
    ax.set_ylabel("Performance")

    return fig, ax


def create_graphs_for_subset(
    source_subset_path: Path, dest_subset_path: Path, stopping_df: pd.DataFrame = None
):
    """Create graphs for the various sets which have been analyzed throughout AL.

    Parameters
    ----------
    source_subset_path : Path
        The path to the subset analysis folder
    dest_subset_path : Path
        A corresponding path to place the created graphs in
    stopping_df : pd.DataFrame, optional
        Stopping results for plotting
    """

    averages = {"macro_avg", "micro_avg", "weighted_avg", "samples_avg"}

    for data_file in sorted(source_subset_path.iterdir()):

        df = pd.read_csv(data_file, index_col=0)

        if data_file.stem in averages:
            y_columns = ["f1-score"]
        elif data_file.stem == "accuracy":
            y_columns = ["accuracy"]
        elif data_file.stem == "hamming_loss":
            y_columns = ["hamming_loss"]
        else:
            # These refer to various individual categories
            y_columns = ["f1-score"]

        fig, ax = plot_from_dataframe(df, x_column="training_data", y_columns=y_columns)

        if stopping_df is not None:
            add_stopping_vlines(fig, ax, stopping_df)

        fig.savefig(dest_subset_path / data_file.with_suffix(".png").name, dpi=400)
        plt.close()
        plt.clf()
        plt.cla()
        fig.clf()


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
        (container.processed_test_set_path, container.graphs_test_set_path),
        (container.processed_train_set_path, container.graphs_train_set_path),
    )
    for source_path, dest_path in paths:
        create_graphs_for_subset(source_path, dest_path, stopping_df)


def main(params: Dict[str, Union[str, int]]) -> None:
    """Run the graphing algorithm for a set of experiment parmameters.

    Parameters
    ----------
    params : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    print("Beginning Graphing", flush=True)

    oh = output_helper.OutputHelper(params)
    # TODO: Once the stopping methods module is working properly, add the stopping vlines again.
    create_graphs_for_container(
        oh.container,
        None,
        False,  # [repr(stopping_methods.StabilizingPredictions())],
    )

    print("Ending Graphing", flush=True)
