"""Average the data across multiple sampling versions.
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict, List, Union

import pandas as pd

import config
import graphing
import output
import stat_helper
import stopping_methods


def average_processed(
    in_containers: List[output.OutputDataContainer], out_container: output.OutputDataContainer
):
    """Average the contents from multiple experiments that have been processed.

    Parameters
    ----------
    in_containers : List[output.OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : output.OutputDataContainer
        The average output container
    """

    for filename in ("accuracy.csv", "macro_avg.csv", "weighted_avg.csv"):
        files = [c.processed_stop_set_all / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_stop_set_all / filename)

        files = [c.processed_train_set_all / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_train_set_all / filename)

        files = [c.processed_test_set_all / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_test_set_all / filename)


def average_stopping(
    in_containers: List[output.OutputDataContainer], out_container: output.OutputDataContainer
):
    """Average the stopping results from multiple experiments.

    Parameters
    ----------
    in_containers : List[output.OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : output.OutputDataContainer
        The average output container
    """

    stopping_dfs = [pd.read_csv(c.stopping_results_csv_file, index_col=0) for c in in_containers]
    mean_stopping_df = stat_helper.mean_dataframes(stopping_dfs)
    mean_stopping_df.to_csv(out_container.stopping_results_csv_file)


def average_container(
    in_containers: List[output.OutputDataContainer], out_container: output.OutputDataContainer
):
    """Average the results from multiple experiments.

    Parameters
    ----------
    in_containers : List[output.OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : output.OutputDataContainer
        The average output container
    """

    average_stopping(in_containers, out_container)
    average_processed(in_containers, out_container)


def main(experiment_parameters: Dict[str, Union[str, int]]) -> None:
    """Run the averaging algorithm for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    print("Beginning Averaging", flush=True)

    roh = output.RStatesOutputHelper(output.OutputHelper(experiment_parameters))
    average_container(roh.ind_rstates_containers, roh.avg_rstates_container)
    graphing.create_graphs_for_container(
        roh.avg_rstates_container, [repr(stopping_methods.StabilizingPredictions())]
    )

    print("Ending Averaging", flush=True)


if __name__ == "__main__":

    main(config.experiment_parameters)
