"""Main program to execute processed for a specific set of experiment parameters.
"""

import json
from pprint import pprint
import warnings

from sklearn.exceptions import ConvergenceWarning

import active_learner
import analysis
import averager
import stopping
import verify

def main(config_file=None, experiment_parameters=None, flags=None):
    
    if config_file is None and experiment_parameters is None:
        raise ValueError("One of config_file or experiment_parameters must not be None")
        
    if experiment_parameters is None:
        with open(config_file, 'r') as f:
            experiment_parameters = json.load(f)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if "active_learning" in flags or flags is None:
            active_learner.main(experiment_parameters)
        if "analysis" in flags or flags is None:
            analysis.main(experiment_parameters)
        if "stopping" in flags or flags is None:
            stopping.main(experiment_parameters)
        if "averaging" in flags or flags is None:
            averager.main(experiment_parameters)
        if "verify" in flags or flags is None:
            verify.main(experiment_parameters)
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "/home/hpc/kurlanl1/bloodgood/modAL/output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "initial_pool_size": 10,
        "batch_size": 10, 
        "estimator": "mlp",
        "dataset": "Avila",
        "random_seed": 0,
    }
    
    main(experiment_parameters=experiment_parameters)