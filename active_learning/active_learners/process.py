"""Analyze the raw output from active learning and process into useful data files.

TODO
----
-

FIXME
-----
-
"""

from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import

import json
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from active_learning.active_learners import output_helper


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
            if isinstance(v, dict):
                if k not in data:
                    data[k] = {}
                for k2, v2 in v.items():
                    if k2 in data[k]:
                        data[k][k2].append(v2)
                    else:
                        data[k][k2] = [v2]
            else:
                if k in data:
                    data[k].append(v)
                else:
                    data[k] = [v]

    return data


def process_container(container: output_helper.OutputDataContainer):
    """Process the raw data in a particular container.

    Parameters
    ----------
    container : output_helper.OutputDataContainer
        The container where the raw data is located and where the processed data should go
    """

    paths = (
        (container.raw_stop_set_path, container.processed_stop_set_path),
        (container.raw_test_set_path, container.processed_test_set_path),
        (container.raw_train_set_path, container.processed_train_set_path),
    )
    for raw_path, processed_path in paths:
        reports = list(raw_path.iterdir())
        data = report_jsons_to_dicts(reports)
        dfs = {k.replace(" ", "_"): pd.DataFrame(v) for k, v in data.items()}

        dfs["training_data"] = dfs["training_data"].rename(columns={0: "training_data"})
        dfs["training_data"].insert(0, "iteration", dfs["iteration"][0])
        del dfs["iteration"]

        for k, df in dfs.items():
            if k == "training_data":
                continue
            df.insert(0, "training_data", dfs["training_data"]["training_data"])
            df.insert(0, "iteration", dfs["training_data"]["iteration"])

        dfs["accuracy"] = dfs["accuracy"].rename(columns={0: "accuracy"})
        dfs["hamming_loss"] = dfs["hamming_loss"].rename(columns={0: "hamming_loss"})

        for k, df in dfs.items():
            if k == "training_data":
                continue
            df.to_csv(processed_path / f"{k}.csv")
    dfs["training_data"].to_csv(container.training_data_file)


def main(experiment_parameters: Dict[str, Union[str, int]]):
    """Process the raw data from an AL experiment for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """

    print("Beginning Processing", flush=True)

    oh = output_helper.OutputHelper(experiment_parameters)
    process_container(oh.container)

    print("Ending Processing", flush=True)
