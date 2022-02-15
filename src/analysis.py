"""Analyze the raw output from active learning and process into useful data files.
"""

import json
from pathlib import Path
from pprint import pprint

import pandas as pd

from output_helper import OutputHelper

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

def dict_of_dfs_to_csvs(output_helper, dfs):
    
    for k, df in dfs.items():
        if k not in {"accuracy", "macro avg", "weighted avg"}:
            path = output_helper.processed_individual_path / f"{k}.csv"
        elif k == "accuracy":
            df.rename(columns={0:"accuracy"}, inplace=True)
            path = output_helper.processed_average_path / f"{k}.csv"
        else:
            path = output_helper.processed_average_path / f"{k.replace(' ', '_')}.csv"
        df.to_csv(path)

def main(experiment_parameters=None):
    
    output_helper = OutputHelper(**experiment_parameters)
    report_paths = list(output_helper.report_path.iterdir())
    data = report_jsons_to_dicts(report_paths)
    dfs = report_dicts_to_dfs(data)
    dict_of_dfs_to_csvs(output_helper, dfs)
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "/home/hpc/kurlanl1/bloodgood/modAL/output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "initial_pool_size": 10,
        "batch_size": 10, 
        "estimator": "RandomForestClassifier()",
        #"estimator": "MLPClassifier()",
        #"estimator": "SVC(kernel='linear', probability=True)",
        "dataset": "Avila",
        "random_seed": 0
    }
    
    main(experiment_parameters)