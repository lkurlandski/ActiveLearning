"""Average the data across multiple sampling versions.

TODO
----
- Implement systems to average experiments across different datasets or feature representations.

FIXME
-----
-
"""

from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict, List, Union

import pandas as pd

from active_learning import graphing
from active_learning import output_helper
from active_learning import stat_helper
from active_learning import stopping_methods


def average_processed(
    in_containers: List[output_helper.OutputDataContainer],
    out_container: output_helper.OutputDataContainer,
):
    """Average the contents from multiple experiments that have been processed.

    Parameters
    ----------
    in_containers : List[output_helper.OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : output_helper.OutputDataContainer
        The average output container
    """

    files_to_average = [p.name for p in in_containers[0].processed_train_set_path.glob("*.csv")]
    for filename in files_to_average:
        files = [c.processed_stop_set_path / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_stop_set_path / filename)

        files = [c.processed_train_set_path / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_train_set_path / filename)

        files = [c.processed_test_set_path / filename for c in in_containers]
        mean_df = stat_helper.mean_dataframes_from_files(files, index_col=0)
        mean_df.to_csv(out_container.processed_test_set_path / filename)


def average_stopping(
    in_containers: List[output_helper.OutputDataContainer],
    out_container: output_helper.OutputDataContainer,
):
    """Average the stopping results from multiple experiments.

    Parameters
    ----------
    in_containers : List[output_helper.OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : output_helper.OutputDataContainer
        The average output container
    """

    stopping_files = [c.stopping_results_csv_file for c in in_containers]
    files_exist = [file.exists() for file in stopping_files]
    if not any(files_exist):
        return
    if not all(files_exist):
        raise FileNotFoundError(f"Need all stopping files to exist:\n{pformat(stopping_files)}")

    stopping_dfs = [pd.read_csv(c.stopping_results_csv_file, index_col=0) for c in in_containers]
    mean_stopping_df = stat_helper.mean_dataframes(stopping_dfs)
    mean_stopping_df.to_csv(out_container.stopping_results_csv_file)


def average_container(
    in_containers: List[output_helper.OutputDataContainer],
    out_container: output_helper.OutputDataContainer,
):
    """Average the results from multiple experiments.

    Parameters
    ----------
    in_containers : List[output_helper.OutputDataContainer]
        Data containers from experiments from multiple random states
    out_container : output_helper.OutputDataContainer
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

    roh = output_helper.RStatesOutputHelper(output_helper.OutputHelper(experiment_parameters))
    average_container(roh.ind_rstates_containers, roh.avg_rstates_container)

    graphing.create_graphs_for_container(
        roh.avg_rstates_container,
        None,
        False,  # [repr(stopping_methods.StabilizingPredictions())],
    )

    print("Ending Averaging", flush=True)


if __name__ == "__main__":

    from active_learning import local

    main(local.experiment_parameters)
