"""
Global configurations.
"""

from pathlib import Path

# Location of the template slurm path used to produce the slurm scripts
slurm_template_path = Path("./slurm_template.sh")
# Line number to replace with a custom call to our functions
line_no = 37

# Location of where the configuration files will be written to
config_file_path = Path("./config_files")

# Location to put the slurm scripts
slurm_scripts_path = Path("./slurm/scripts")