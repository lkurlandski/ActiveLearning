"""Main program to execute processed for a specific set of experiment parameters.
"""

import json
from pprint import pprint
import warnings

from sklearn.exceptions import ConvergenceWarning

import active_learner
import processor
import averager
import graphing
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
        if "processor" in flags or flags is None:
            processor.main(experiment_parameters)
        if "stopping" in flags or flags is None:
            stopping.main(experiment_parameters)
            
        # TODO: SLURM overhaul: these should only be run once per set of experiments
        if "averaging" in flags or flags is None:
            averager.main(experiment_parameters)
        if "graphing" in flags or flags is None:
            graphing.main(experiment_parameters)
        if "verify" in flags or flags is None:
            verify.main(experiment_parameters)
    
if __name__ == "__main__":
    
    experiment_parameters = {
        "output_root": "./output",
        "task": "preprocessedClassification",
        "stop_set_size": 1000,
        "batch_size": 10, 
        "estimator": "mlp",
        "dataset": "Avila",
        "random_state": 0,
    }
    
    main(experiment_parameters=experiment_parameters)