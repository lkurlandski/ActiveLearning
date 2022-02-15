"""Rudimentary stopping algorithms.

# TODO: ultimately, we will want to incorporate the stopping criterion to run during AL.
"""

from pprint import pprint
from statistics import mean

import numpy as np
import pandas as pd

from output_helper import OutputHelper

def stabilizing_predictions(output_helper, windows=3, threshold=0.99):
    
    kappas = np.loadtxt(output_helper.kappa_file.as_posix())
    for i in range(len(kappas)):
        if i - windows < 0:
            continue
        
        avg = mean(kappas[i-windows:i])
        
        if avg > threshold:
            return i
        
    # TODO: when the notation of the csv files is refined, this might need be modified.
    return len(kappas) - 2

def main(experiment_parameters):
    
    output_helper = OutputHelper(**experiment_parameters)
    stopping = {
        "stabilizing_predictions": [stabilizing_predictions(output_helper, 3, 0.99)]
    }
    
    accuracy_df = pd.read_csv(output_helper.processed_accuracy_file)
    macro_avg_df = pd.read_csv(output_helper.processed_macro_avg_file)
    weighted_avg_df = pd.read_csv(output_helper.processed_weighted_avg_file)
    
    for k in stopping:
        stop = stopping[k][0]
        accuracy = accuracy_df.at[stop, "accuracy"]
        macro_f1 = macro_avg_df.at[stop, "f1-score"]
        weighted_f1 = weighted_avg_df.at[stop, "f1-score"]
        stopping[k].extend([accuracy, macro_f1, weighted_f1])
        
    df = pd.DataFrame(stopping, index=["annotations", "accuracy", "macro_f1", "weighted_f1"])
    df.to_csv(output_helper.stopping_results_file)
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "/home/hpc/kurlanl1/bloodgood/modAL/output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "initial_pool_size": 10,
        "batch_size": 10, 
        "estimator": "mlp",
        "dataset": "Avila",
        "random_seed": 5
    }
    
    main(experiment_parameters)