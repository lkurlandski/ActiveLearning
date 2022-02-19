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