"""Analyze the raw output from active learning and process into useful data files.
"""

import json
from pathlib import Path
from pprint import pprint

import pandas as pd

import output_helper

def report_jsons_to_dicts(paths):
    
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

def report_dicts_to_dfs(data):
    
    dfs = {}
    for k in list(data.keys()):
        dfs[k] = pd.DataFrame(data[k])
        
    return dfs

def dict_of_dfs_to_csvs(oh, dfs, subset):
    
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

def main(experiment_parameters=None):
    
    oh = output_helper.OutputHelper(experiment_parameters)
    
    for subset in ("test", "train", "stop_set"):
        report_paths = list(oh.ind_rstates_paths[f'report_{subset}_path'].iterdir())
        data = report_jsons_to_dicts(report_paths)
        dfs = report_dicts_to_dfs(data)
        dict_of_dfs_to_csvs(oh, dfs, subset)
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 1,
        "estimator": "svm",
        "dataset": "Iris",
        "random_state": 0,
    }
    
    main(experiment_parameters)