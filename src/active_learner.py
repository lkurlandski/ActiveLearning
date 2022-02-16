"""Run the active learning process.
"""

import json
import math
from pathlib import Path
from pprint import pprint
from random import choices
import warnings

import matplotlib.pyplot as plt
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
import vectorizers

def print_pool_update(
        unlabeled_pool_initial_size,
        y_training,
        y_pool,
        iteration,
        n_iterations
    ):
    
    print(
        f"Iteration {iteration} / {n_iterations} = {round(100 * iteration/n_iterations, 2)}%:"
        f"\t|U_0|:{unlabeled_pool_initial_size}"
        f"\t|U|:{y_pool.shape[0]}"
        f"\t|L|:{y_training.shape[0]}",
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
    initial_pool_size = int(experiment_parameters["initial_pool_size"])
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

    # Add additional random ids if the required size not met
    extra_examples_needed = initial_pool_size - len(idx)
    if extra_examples_needed > 0:
        idx += choices([i for i in range(len(y_train)) if i not in idx], k=extra_examples_needed)

    # Remove first pool from the train set
    X_pool, y_pool = X_train.copy(), y_train.copy()
    X_initial, y_initial = X_pool[idx], y_pool[idx]
    X_pool, y_pool = stat_helper.remove_ids_from_array(X_pool, idx), stat_helper.remove_ids_from_array(y_pool, idx)
    
    # Number of AL iterations
    n_iterations = math.ceil(y_pool.shape[0] / batch_size) + 1

    # Select the stop set
    stop_set_idx = choices([i for i in range(len(y_train))], k=stop_set_size)   

    # Create the active learner
    learner = ActiveLearner(
        estimator=estimator,
        query_strategy=uncertainty_batch_sampling,
        X_training=X_initial, 
        y_training=y_initial
    )
    
    # Display the current pool sizes
    print_pool_update(unlabeled_pool_initial_size, learner.y_training, y_pool, 0, n_iterations)
    
    # First evaluation of the learner
    predictions = learner.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1, output_dict=True)
    with open(oh.report_test_path / f"1.json", 'w') as f:
        json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

    # Set up Stabilizing Predictions tracking
    previous_stop_set_predictions = learner.predict(X_train[stop_set_idx])
    kappas = [np.NaN]
    
    for i in range(1, n_iterations + 1):

        # Retrain the learner
        query_idx, query_sample = learner.query(X_pool=X_pool, n_instances=batch_size)
        query_labels = y_pool[query_idx]
        learner.teach(query_sample, query_labels)
        
        # Print the progress of the learning procedure
        print_pool_update(unlabeled_pool_initial_size, learner.y_training, y_pool, i, n_iterations)

        # Evaluate the learner on the test set
        predictions = learner.predict(X_test)
        report = classification_report(y_test, predictions, zero_division=1, output_dict=True)
        with open(oh.report_test_path / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

        # Evaluate the learner on the train set
        predictions = learner.predict(X_train)
        report = classification_report(y_train, predictions, zero_division=1, output_dict=True)
        with open(oh.report_train_path / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))
            
        # Evaluate the learner on the stop set
        stop_set_predictions = learner.predict(X_train[stop_set_idx])
        report = classification_report(y_train[stop_set_idx], stop_set_predictions, zero_division=1, output_dict=True)
        with open(oh.report_stopset_path / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))
        
        # Update the list of kappas for stabilizing predictions
        kappa = cohen_kappa_score(stop_set_predictions, previous_stop_set_predictions)
        kappas.append(kappa)
        previous_stop_set_predictions = stop_set_predictions
        
    np.savetxt(oh.kappa_file, np.array(kappas), fmt='%f', delimiter=',')
    
if __name__ == "__main__":
    
    experiment_parameters = {
        # Only one value permitted
        "output_root": "/home/hpc/kurlanl1/bloodgood/modAL/output",
        "task": "preprocessedClassification",
        # Iterable of values required
        "stop_set_size": 1000,
        "initial_pool_size": 10,
        "batch_size": 100,
        "estimator": "svm-ova",   # ["mlp", "svm", "svm-ova", "rf"],
        "dataset": "20NewsGroups-raw",
        "random_state": 0,
    }
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(experiment_parameters)