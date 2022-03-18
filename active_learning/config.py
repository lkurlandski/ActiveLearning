"""
Global configurations.
"""

from pathlib import Path

# This codebase needs to know where the ActiveLearning directory is
location_of_ActiveLearning_dir = Path().resolve().as_posix()

# Location of the template slurm path used to produce the slurm scripts
slurm_template_path = Path("./slurm_template.sh")

# Location of where the configuration files will be written to
config_file_path = Path("./config_files")

# Location to put the slurm scripts
slurm_scripts_path = Path("./slurm/scripts")

# A convienient set of experiment paramaters for testing individual aspects of the system
experiment_parameters = {
    "output_root": "./outputGarbage",
    "task": "cls",
    "stop_set_size": 1000,
    "batch_size": 7,
    "base_learner": "SVC",
    "multiclass": "ovr",
    "feature_representation": "preprocessed",
    "dataset": "Iris",
    "random_state": 0,
}
