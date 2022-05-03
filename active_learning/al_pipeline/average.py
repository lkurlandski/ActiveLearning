"""Average the data across multiple sampling versions.
"""

import datetime
from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import time
from typing import List

import pandas as pd

from active_learning.al_pipeline import graph
from active_learning.al_pipeline.helpers import (
    IndividualOutputDataContainer,
    OutputDataContainer,
    OutputHelper,
    RStatesOutputHelper,
    Params,
)
from active_learning import stat_helper


def average_processed(
    in_containers: List[OutputDataContainer],
    out_container: OutputDataContainer,
):
    """Average the contents from multiple experiments that have been processed.

    Parameters
    ----------
    in_containers : List[OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : OutputDataContainer
        The average output container
    """

    files_to_average = [p.name for p in in_containers[0].processed_train_set_path.glob("*.csv")]
    for filename in files_to_average:

        files = [c.processed_train_set_path / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_train_set_path / filename)

        files = [c.processed_test_set_path / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_test_set_path / filename)


def average_stopping(
    in_containers: List[OutputDataContainer],
    out_container: OutputDataContainer,
):
    """Average the stopping results from multiple experiments.

    Parameters
    ----------
    in_containers : List[OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : OutputDataContainer
        The average output container
    """

    stopping_files = [c.stopping_results_file for c in in_containers]
    files_exist = [file.exists() for file in stopping_files]
    if not any(files_exist):
        return
    if not all(files_exist):
        raise FileNotFoundError(f"Need all stopping files to exist:\n{pformat(stopping_files)}")

    stopping_dfs = [pd.read_csv(c.stopping_results_file, index_col=0) for c in in_containers]
    mean_stopping_df = stat_helper.mean_dataframes(stopping_dfs)
    mean_stopping_df.to_csv(out_container.stopping_results_file)


def average_container(
    in_containers: List[IndividualOutputDataContainer],
    out_container: OutputDataContainer,
):
    """Average the results from multiple experiments.

    Parameters
    ----------
    in_containers : List[IndividualOutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : OutputDataContainer
        The average output container
    """
    print("-" * 80, flush=True)
    start = time.time()
    print("0:00:00 -- Starting Averaging", flush=True)

    average_stopping(in_containers, out_container)
    average_processed(in_containers, out_container)

    diff = datetime.timedelta(seconds=(round(time.time() - start)))
    print(f"{diff} -- Ending Averaging", flush=True)
    print("-" * 80, flush=True)


def main(params: Params) -> None:
    """Run the averaging algorithm for a set of experiment parmameters.

    Parameters
    ----------
    params : Params
        Epxperimental parameters.
    """

    if params["early_stop_mode"] != "none":
        raise Exception(
            "Averaging across random states when and early mode was enabled not supported...yet."
        )

    roh = RStatesOutputHelper(OutputHelper(params))
    average_container(roh.ind_rstates_containers, roh.avg_rstates_container)

    graph.create_graphs_for_container(
        roh.avg_rstates_container,
        None,
    )
