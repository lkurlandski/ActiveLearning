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

# TODO: avoid the from x import y format for local code because it is confusing for determining 
    # which functions are in the current file and which are not
from estimators import get_estimator_from_code
from input_helper import get_dataset
from output_helper import OutputHelper
from stat_helper import remove_ids_from_array
import vectorizers

# TODO: currently, this runs the experiments on the hpc nodes, but writes files locally.
    # Develop a system (along with the output helper) that writes the experiment
    # to the /local/scratch directory on the node, then copies them over to the local directory 
    # when complete.

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
    estimator = get_estimator_from_code(experiment_parameters["estimator"])
    random_seed = int(experiment_parameters["random_seed"])

    # TODO: come up with a more efficient and elegant way to handle the data splits,
        # especially when selecting and removing the very first pool
    # Get the dataset
    X_train, X_test, y_train, y_test, labels = \
        get_dataset(experiment_parameters["dataset"], random_seed)
    unlabeled_pool_initial_size = y_train.shape[0]
    
    # Setup output directory structure
    output_helper = OutputHelper(**experiment_parameters)
    output_helper.setup_output_path(remove_existing=True)
    print(f"output_helper.output_path:{output_helper.output_path.as_posix()}\n")

    # Get ids of one instance of each class
    idx = get_index_for_each_class(y_train, labels)

    # Add additional random ids if the required size not met
    extra_examples_needed = initial_pool_size - len(idx)
    if extra_examples_needed > 0:
        idx += choices([i for i in range(len(y_train)) if i not in idx], k=extra_examples_needed)

    # Remove first pool from the train set
    X_pool, y_pool = X_train.copy(), y_train.copy()
    X_initial, y_initial = X_pool[idx], y_pool[idx]
    X_pool, y_pool = remove_ids_from_array(X_pool, idx), remove_ids_from_array(y_pool, idx)
    
    # Number of AL iterations
    n_iterations = math.ceil(y_pool.shape[0] / batch_size) + 1

    # Select the stop set
    stop_set_idx = choices([i for i in range(len(y_train))], k=stop_set_size)
    
    # TODO: develop and elegant way of using pipelines. 
        # Perhaps wrap every estimator in a Pipeline by default?
        # Then apply an appropriate procedure of transformations specified by a new parameter.
    
    # Wrap a pipeline around the estimator!
    #from sklearn.pipeline import Pipeline
    #from transformers import BertTokenizer, BertModel
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #bert_model = BertModel.from_pretrained("bert-base-uncased")
    #bert_transformer = vectorizers.BertTransformer(tokenizer, bert_model)
    #Pipeline(
    #    [
    #        ("vectorizer", bert_transformer),
    #        ("estimator", estimator),
    #    ]
    #)    

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
    with open(output_helper.report_path / f"1.json", 'w') as f:
        json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))

    # Set up Stabilizing Predictions tracking
    previous_stop_set_predictions = learner.predict(X_train[stop_set_idx])
    kappas = [np.NaN]
    
    for i in range(1, n_iterations + 1):

        # Retrain the learner
        query_idx, query_sample = learner.query(X_pool=X_pool, n_instances=batch_size)
        query_labels = y_pool[query_idx]
        learner.teach(query_sample, query_labels)
        
        print_pool_update(unlabeled_pool_initial_size, learner.y_training, y_pool, i, n_iterations)

        # Evaluate the learner on the test set and stop set
        predictions = learner.predict(X_test)
        report = classification_report(y_test, predictions, zero_division=1, output_dict=True)
        with open(output_helper.report_path / f"{str(i)}.json", 'w') as f:
            json.dump(report, f, sort_keys=True, indent=4, separators=(',', ': '))
        
        # Track Stabilizing Predictions stopping methods
        stop_set_predictions = learner.predict(X_train[stop_set_idx])
        kappa = cohen_kappa_score(stop_set_predictions, previous_stop_set_predictions)
        kappas.append(kappa)
        previous_stop_set_predictions = stop_set_predictions
        
    np.savetxt(output_helper.kappa_file, np.array(kappas), fmt='%f', delimiter=',')
    
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
        "random_seed": 0,
    }
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main(experiment_parameters)
