"""Analyze the raw output from active learning and process into useful data files.
"""

import json
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Union

import pandas as pd

import output_helper

def report_jsons_to_dicts(
        paths:List[Path]
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
    for p in sorted(paths):
        with open(p, 'r') as f:
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

def report_dicts_to_dfs(
        data:Dict[str, Union[List[float], Dict[str, List[float]]]]
    ) -> Dict[str, pd.DataFrame]:
    """Convert the report.json dictionary representations into dataframe form.

    Parameters
    ----------
    data : Dict[str, Union[List[float], Dict[str, List[float]]]]
        report.json dictionary representation

    Returns
    -------
    Dict[str, pd.DataFrame]
        Named dataframes of data
    """
    
    dfs = {}
    for k in list(data.keys()):
        dfs[k] = pd.DataFrame(data[k])
        
    return dfs

def dict_of_dfs_to_csvs(
        oh:output_helper.OutputHelper, 
        dfs:Dict[str, pd.DataFrame], 
        subset:str
    ) -> None:
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
        if k not in {"accuracy", "macro avg", "weighted avg"}:
            path = oh.ind_rstates_paths[f"processed_{subset}_ind_path"] / f"{k}.csv"
        elif k == "accuracy":
            df.rename(columns={0:"accuracy"}, inplace=True)
            path = oh.ind_rstates_paths[f"processed_{subset}_avg_path"] / f"{k}.csv"
        else:
            path = \
                oh.ind_rstates_paths[f"processed_{subset}_avg_path"] / f"{k.replace(' ', '_')}.csv"
        df.to_csv(path)

def main(experiment_parameters:Dict[str, Union[str, int]]):
    """Process the raw data from an AL experiment for a set of experiment parmameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        A single set of hyperparmaters and for the active learning experiment.
    """
    
    oh = output_helper.OutputHelper(experiment_parameters)
    
    num_training_data = pd.read_csv(oh.ind_rstates_paths["num_training_data_file"])['training_data']
    
    for subset in ("test", "train", "stop_set"):
        report_paths = list(oh.ind_rstates_paths[f'report_{subset}_path'].iterdir())
        data = report_jsons_to_dicts(report_paths)
        dfs = report_dicts_to_dfs(data)
        for df in dfs.values():
            df.insert(0, 'training_data', num_training_data)
        dict_of_dfs_to_csvs(oh, dfs, subset)
    
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