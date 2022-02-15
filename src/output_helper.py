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
    # TODO: remove the initial_pool_size parameter
    # TODO: rename the random_seed parameter to random_state
    
    def __init__(self, 
            output_root, 
            task, 
            initial_pool_size, 
            stop_set_size, 
            batch_size, 
            estimator, 
            dataset,
            random_seed
        ):
        
        self.output_root = Path(output_root)
        
        self.dataset_path = self.output_root / str(task) / str(initial_pool_size) / str(stop_set_size) / str(batch_size) / str(estimator) / str(dataset)
        
        # Individual seed paths
        self.ind_seeds_path = self.dataset_path / "ind_seeds"
        self.output_path = self.ind_seeds_path / str(random_seed)
        
        self.raw_path = self.output_path / "raw"
        self.report_path = self.raw_path / "reports"
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
        self.report_path.mkdir(exist_ok=exists_ok)
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