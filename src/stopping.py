"""Rudimentary stopping algorithms.
"""

from pprint import pprint
from statistics import mean

import numpy as np
import pandas as pd

import output_helper

def stabilizing_predictions(kappas, windows=3, threshold=0.99):
    
    for i in range(len(kappas)):
        if i - windows < 0:
            continue
        
        avg = mean(kappas[i-windows:i])
        
        if avg > threshold:
            return i
        
    # TODO: when the notation of the csv files is refined, this might need be modified.
    return len(kappas) - 2

def main(experiment_parameters):
    
    oh = output_helper.OutputHelper(experiment_parameters)
    
    kappas = np.loadtxt(oh.ind_rstates_paths["kappa_file"].as_posix())
    
    stopping = {
        "stabilizing_predictions": [stabilizing_predictions(kappas, 3, 0.99)]
    }
    
    num_training_data = pd.read_csv(oh.ind_rstates_paths["num_training_data_file"])
    accuracy_df = pd.read_csv(oh.ind_rstates_paths["processed_test_accuracy_path"])
    macro_avg_df = pd.read_csv(oh.ind_rstates_paths["processed_test_macro_avg_path"])
    weighted_avg_df = pd.read_csv(oh.ind_rstates_paths["processed_test_weighted_avg_path"])
    
    for k in stopping:
        stop = stopping[k][0]
        annotations = num_training_data.at[stop, "training_data"]
        accuracy = accuracy_df.at[stop, "accuracy"]
        macro_f1 = macro_avg_df.at[stop, "f1-score"]
        weighted_f1 = weighted_avg_df.at[stop, "f1-score"]
        stopping[k].extend([annotations, accuracy, macro_f1, weighted_f1])
        
    df = pd.DataFrame(stopping, index=["iteration", "annotations", "accuracy", "macro_f1", "weighted_f1"])
    df.to_csv(oh.ind_rstates_paths["stopping_results_file"])
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 7,
        "estimator": "svm",
        "dataset": "Iris",
        "random_state": 0,
    }
    
    main(experiment_parameters)