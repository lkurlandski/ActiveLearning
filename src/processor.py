"""Analyze the raw output from active learning and process into useful data files.
"""

import json
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Dict, List, Union

import pandas as pd

import output


def report_jsons_to_dicts(
    paths: List[Path],
) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
    """Convert sroted report.json files into a dictionary strcuture.

    Parameters
    ----------
    paths : List[Path]
        List of files to convert

    Returns
    -------
    Dict[str, Union[List[float], Dict[str, List[float]]]]
        The named data elements found along from each report.json file concatonated into lists
    """

    data = {}
    for p in sorted(paths, key=lambda x: int(x.name.replace("report", "").replace(".json", ""))):
        with open(p, "r", encoding="utf8") as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, float):
                if k in data:
                    data[k].append(v)
                else:
                    data[k] = [v]
            elif isinstance(v, dict):
                if k not in data:
                    data[k] = {}
                for k2, v2 in v.items():
                    if k2 in data[k]:
                        data[k][k2].append(v2)
                    else:
                        data[k][k2] = [v2]

    return data


def dict_of_dfs_to_csvs(dfs: Dict[str, pd.DataFrame], processed_path: Path) -> None:
    """Convert the processed dataframes into csv files.

    Parameters
    ----------
    oh : output_helper.OutputHelper
        OutputHelper for this experiment
    dfs : Dict[str, pd.DataFrame]
        Named dataframes of data
    subset : str
        one of train, test, or stop_set to refer to a particular data location
    """

    for k, df in dfs.items():
        if k in {"accuracy", "macro_avg", "weighted_avg"}:
            path = processed_path / output.OutputDataContainer.overall_str / f"{k}.csv"
        else:
            path = processed_path / output.OutputDataContainer.ind_cat_str / f"{k}.csv"
        df.to_csv(path)


def process_container(container: output.OutputDataContainer):
    """Process the raw data in a particular container.

    Parameters
    ----------
    container : output.OutputDataContainer
        The container where the raw data is located and where the processed data should go
    """

    # Keeping this as a pd.Series allows for seemless application to dataframe with different length
    num_training_data = pd.read_csv(container.training_data_file)["training_data"]

    paths = (
        (container.raw_stop_set_path, container.processed_stop_set_path),
        (container.raw_test_set_path, container.processed_test_set_path),
        (container.raw_train_set_path, container.processed_train_set_path),
    )
    for raw_path, processed_path in paths:
        reports = list(raw_path.iterdir())
        data = report_jsons_to_dicts(reports)
        dfs = {k.replace(" ", "_"): pd.DataFrame(v) for k, v in data.items()}
        for df in dfs.values():
            df.insert(0, "training_data", num_training_data)
        dfs["accuracy"] = dfs["accuracy"].rename(columns={0: "accuracy"})
        dict_of_dfs_to_csvs(dfs, processed_path)


def main(experiment_parameters: Dict[str, Union[str, int]]):
    """Process the raw data from an AL experiment for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    print("Beginning Processing", flush=True)

    oh = output.OutputHelper(experiment_parameters)
    process_container(oh.container)

    print("Ending Processing", flush=True)


if __name__ == "__main__":

    main(
        experiment_parameters={
            "output_root": "./output",
            "task": "cls",
            "stop_set_size": 1000,
            "batch_size": 10,
            "estimator": "mlp",
            "dataset": "Iris",
            "random_state": 0,
        }
    )
