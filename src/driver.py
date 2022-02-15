"""From experiment paramaters, create config files, then slurm scripts, and sbatch them.
"""

import json
from pprint import pprint
import shutil
import subprocess

import config
import main

# TODO: implement a more elegant way of constructing the slurm scripts
# TODO: figure out a way to make the slurm scripts get printed to dynamically
# TODO: use job arrays in the slurm scripts, then store each array in its own folder
def sbatch_config_files(flags):
    
    # Delete old files
    if config.slurm_scripts_path.exists():
        shutil.rmtree(config.slurm_scripts_path)
    config.slurm_scripts_path.mkdir()
    
    with open(config.slurm_template_path, 'r') as f:
        slurm_lines = f.readlines()
    
    for i, p in enumerate(sorted(config.config_file_path.glob("*.json"))):
        
        slurm_lines[config.line_no] = f"main.main(config_file='{p.as_posix()}', flags={flags})"
        
        slurm_script_file = config.slurm_scripts_path / f"{i}.sh"
        with open(slurm_script_file, 'w') as f:
            f.writelines(slurm_lines)
            
        result = subprocess.run(["sbatch", slurm_script_file.as_posix()], capture_output=True)
        print(result.stdout)
        
def create_config_files(experiment_parameters_lists, flags, local):
    
    # Delete old files
    if config.config_file_path.exists():
        shutil.rmtree(config.config_file_path)
    config.config_file_path.mkdir()
    
    i = 0
    
    for stop_set_size in experiment_parameters_lists["stop_set_size"]:
        for batch_size in experiment_parameters_lists["batch_size"]:
            for initial_pool_size in experiment_parameters_lists["initial_pool_size"]:
                for estimator in experiment_parameters_lists["estimator"]:
                    for dataset in experiment_parameters_lists["dataset"]:
                        for random_seed in experiment_parameters_lists["random_seed"]:
                            experiment_parameters = {
                                "output_root": experiment_parameters_lists["output_root"],
                                "task": experiment_parameters_lists["task"],
                                "stop_set_size": stop_set_size,
                                "initial_pool_size": initial_pool_size,
                                "batch_size": batch_size, 
                                "estimator": estimator,
                                "dataset": dataset,
                                "random_seed": random_seed
                            }
                            experiment_parameters = {k : str(v) for k, v in experiment_parameters.items()}
                            
                            if local:
                                main.main(experiment_parameters=experiment_parameters, flags=flags)
                            else:
                                config_file = config.config_file_path / f"{i}.json"
                                with open(config_file, 'w') as f:
                                    json.dump(experiment_parameters, f, sort_keys=True, indent=4, separators=(',', ': '))
                                i += 1