"""Assist with setting up and maintaining the output structure for experiments.
"""

from pathlib import Path
import shutil

# TODO: currently, this runs the experiments on the hpc nodes, but writes files locally.
    # Develop a system (along with the output helper) that writes the experiment
    # to the /local/scratch directory on the node, then copies them over to the local directory 
    # when complete.

def contains_data(path, ignore_raw=False):
    
    files = set((p.name for p in path.glob("*")))

    if "processed" in files and "stopping" in files:
        if ignore_raw:
            return True
        if "raw" in files:
            return True
    
    return False

class OutputHelper:
    
    # TODO: create class variable that describe the relative location of various paths/files,
        # e.g., stopping/results.csv relative to its parent directory
    # TODO: remove the initial_pool_size parameter (should simply be one element from distinct classes)
    # TODO: come up with a more comprehensive naming scheme when we figure out exactly what data we 
        # want to record
        
    required_experiment_parameters = {
        "output_root",
        "task",
        "stop_set_size",
        "initial_pool_size",
        "batch_size",
        "estimator",
        "dataset",
        "random_state"
    }
    
    def __init__(self, experiment_parameters):
        
        if set(experiment_parameters.keys()) != self.required_experiment_parameters:
            raise ValueError(f"Expected the following parameters:\n\t"
                f"{sorted(self.required_experiment_parameters)}\nbut recieved the following:\n\t"
                f"{sorted(experiment_parameters.keys())}"
            )
            
        self.experiment_parameters = experiment_parameters
        e = self.experiment_parameters
        
        self.output_root = Path(e['output_root'])
        
        self.task_path = self.output_root / e['task']
        self.estimator_path = self.task_path / str(e['stop_set_size']) / str(e['batch_size']) / e['estimator']
        self.dataset_path = self.estimator_path / e['dataset']
        
        # Individual seed paths
        self.ind_seeds_path = self.dataset_path / "ind_seeds"
        self.output_path = self.ind_seeds_path / str(e['random_state'])
        
        self.raw_path = self.output_path / "raw"
        self.report_test_path = self.raw_path / "reportsTest"
        self.report_train_path = self.raw_path / "reportsTrain"
        self.report_stopset_path = self.raw_path / "reportsStopset"
        self.kappa_file = self.raw_path / "kappas.txt"
        
        self.processed_path = self.output_path / "processed"
        self.processed_individual_path = self.processed_path / "individual"
        self.processed_average_path = self.processed_path / "average"
        self.processed_accuracy_file = self.processed_average_path / "accuracy.csv"
        self.processed_macro_avg_file = self.processed_average_path / "macro_avg.csv"
        self.processed_weighted_avg_file = self.processed_average_path / "weighted_avg.csv"
        
        self.stopping_path = self.output_path / "stopping"
        self.stopping_results_file = self.stopping_path / "results.csv"
        
        # Average seed paths
        self.avg_seeds_path = self.dataset_path / "avg_seeds"
        self.avg_processed_path = self.avg_seeds_path / "processed"
        self.avg_processed_individual_path = self.avg_processed_path / "individual"
        self.avg_processed_average_path = self.avg_processed_path / "average"
        self.avg_processed_accuracy_file = self.avg_processed_average_path / "accuracy.csv"
        self.avg_processed_macro_avg_file = self.avg_processed_average_path / "macro_avg.csv"
        self.avg_processed_weighted_avg_file = self.avg_processed_average_path / "weighted_avg.csv"

        self.avg_stopping_path = self.avg_seeds_path / "stopping"
        self.avg_stopping_results_file = self.avg_stopping_path / "results.csv"

    def setup_output_path(self, remove_existing=False, exists_ok=True):
        if self.output_path.exists():
            if remove_existing:
                shutil.rmtree(self.output_path)
            elif exists_ok:
                pass
            else:
                raise FileExistsError(f"{self.output_path} exists.")
        
        # Individual seed paths
        self.ind_seeds_path.mkdir(parents=True, exist_ok=exists_ok)
        self.output_path.mkdir(exist_ok=exists_ok)
        self.raw_path.mkdir(exist_ok=exists_ok)
        self.report_test_path.mkdir(exist_ok=exists_ok)
        self.report_train_path.mkdir(exist_ok=exists_ok)
        self.report_stopset_path.mkdir(exist_ok=exists_ok)
        self.processed_path.mkdir(exist_ok=exists_ok)
        self.processed_individual_path.mkdir(exist_ok=exists_ok)
        self.processed_average_path.mkdir(exist_ok=exists_ok)
        self.stopping_path.mkdir(exist_ok=exists_ok)
        
        # Average seed paths
        self.avg_seeds_path.mkdir(exist_ok=exists_ok)
        self.avg_processed_path.mkdir(exist_ok=exists_ok)
        self.avg_processed_individual_path.mkdir(exist_ok=exists_ok)
        self.avg_processed_average_path.mkdir(exist_ok=exists_ok)
        self.avg_stopping_path.mkdir(exist_ok=exists_ok)