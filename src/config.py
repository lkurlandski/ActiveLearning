"""
Global configurations.

TODO: SLURM overhaul
"""

from pathlib import Path

# Location of the template slurm path used to produce the slurm scripts
slurm_template_path = Path("./slurm_template.sh")

location_of_active_learning_dir = "/home/hpc/kurlanl1/bloodgood/ActiveLearning"

# TODO: SLURM overhaul: should be removed
# Line number to replace with a custom call to our functions
line_no = 37

# Location of where the configuration files will be written to
config_file_path = Path("./config_files")

# Location to put the slurm scripts
slurm_scripts_path = Path("./slurm/scripts")