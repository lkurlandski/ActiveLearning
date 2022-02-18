"""Run the active learning process.
"""

import json
import math
from pathlib import Path
from pprint import pprint
from random import choices
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import cohen_kappa_score, classification_report

from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner

import estimators
import input_helper
import output_helper
import stat_helper

def print_pool_update(unlabeled_pool_initial_size, y_pool, iteration, n_iterations):
    
    print(
        f"Iteration {iteration} / {n_iterations-1} = {round(100 * iteration/(n_iterations-1), 2)}%:"
        f"\t|U_0|:{unlabeled_pool_initial_size}"
        f"\t|U|:{y_pool.shape[0]}"
        f"\t|L|:{unlabeled_pool_initial_size - y_pool.shape[0]}",
        flush=True
    )

def get_index_for_each_class(y, labels):
    
    # TODO: come up with a more elegant way to handle label encoding.
    if y[0] not in labels:
        labels = [i for i in range(len(labels))]
        
    idx = {l : None for l in labels}
    for i, l in enumerate(y):
        if idx[l] is None:
            idx[l] = i
    idx = [i for i in idx.values()]
    
    return idx

def main(experiment_parameters):
    
    print("active_learner called with experiment_parameters:")
    pprint(experiment_parameters)

    # Extract hyperparameters from the experiment parameters
    stop_set_size = int(experiment_parameters["stop_set_size"])
    batch_size = int(experiment_parameters["batch_size"])
    # TODO: pipelines
    estimator = estimators.get_estimator_from_code(experiment_parameters["estimator"])
    random_state = int(experiment_parameters["random_state"])

    # TODO: the names of the various sets are a little confusing: train, test, pool, initial etc.
    # Get the dataset
    X_train, X_test, y_train, y_test, labels = \
        input_helper.get_dataset(experiment_parameters["dataset"], random_state)
    unlabeled_pool_initial_size = y_train.shape[0]
    
    # TODO: SLURM overhaul
    # Setup output directory structure
    oh = output_helper.OutputHelper(experiment_parameters)
    oh.setup_output_path(remove_existing=True)
    print(f"output_path:{oh.output_path.as_posix()}\n")

    # Get ids of one instance of each class
    idx = get_index_for_each_class(y_train, labels)

    # Remove first pool from the train set
    X_pool = X_train.copy()
    y_pool = y_train.copy()
    
    # Number of AL iterations
    n_iterations = math.ceil(y_pool.shape[0] / batch_size) + 1
    
    # Write the size of the unlabeled pool at each iteration to a file.
    pd.DataFrame({'training_data' : 
        [len(idx) + i * batch_size for i in range(n_iterations - 1)] + [y_train.shape[0]]
    }).to_csv(oh.ind_rstates_paths['num_training_data_file'])

    # Select the stop set
    stop_set_idx = choices([i for i in range(len(y_train))], k=stop_set_size) 

    # These assets will be initialized in the 0th iteration of AL
    learner, previous_stop_set_predictions, kappas = None, None, None
    
    for i in range(n_iterations):
        
        # Setup the learner and stabilizing predictions in the 0th iteration.
        if i == 0:
            learner = ActiveLearner(estimator=estimator, query_strategy=uncertainty_batch_sampling)
            learner.teach(X_pool[idx], y_pool[idx])
            previous_stop_set_predictions = learner.predict(X_train[stop_set_idx])
            kappas = [np.NaN]
        
        # Retrain the learner
        else:
            idx, query_sample = learner.query(X_pool=X_pool, n_instances=batch_size)
            query_labels = y_pool[idx]
            learner.teach(query_sample, query_labels)
        
        # Remove the queried elements from the unlabeled pool
        X_pool = stat_helper.remove_ids_from_array(X_pool, idx)
        y_pool = stat_helper.remove_ids_from_array(y_pool, idx)
        
        # Print the progress of the learning procedure
        print_pool_update(unlabeled_pool_initial_size, y_pool, i, n_iterations)

        # Evaluate the learner on the test set
        predictions = learner.predict(X_test)
        report = classification_report(
            y_test, predictions, zero_division=1, output_dict=True)
        with open(oh.ind_rstates_paths["report_test_path"] / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

        # Evaluate the learner on the train set
        predictions = learner.predict(X_train)
        report = classification_report(
            y_train, predictions, zero_division=1, output_dict=True)
        with open(oh.ind_rstates_paths["report_train_path"] / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))
            
        # Evaluate the learner on the stop set
        stop_set_predictions = learner.predict(X_train[stop_set_idx])
        report = classification_report(
            y_train[stop_set_idx], stop_set_predictions, zero_division=1, output_dict=True)
        with open(oh.ind_rstates_paths["report_stop_set_path"] / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))
        
        # Update the list of kappas for stabilizing predictions
        kappa = cohen_kappa_score(stop_set_predictions, previous_stop_set_predictions)
        kappas.append(kappa)
        previous_stop_set_predictions = stop_set_predictions
        
    np.savetxt(oh.ind_rstates_paths["kappa_file"], np.array(kappas), fmt='%f', delimiter=',')
    
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
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(experiment_parameters)
