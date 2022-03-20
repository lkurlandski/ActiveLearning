"""From experiment paramaters, create config files, then slurm scripts, and sbatch them.
"""

import json
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import subprocess
from typing import Dict, List, Set, Union

from active_learning import runner


# Location of the ActiveLearning directory
root_path = Path("/home/hpc/kurlanl1/bloodgood/ActiveLearning")

# Interpreter path
python_path = root_path / "env/bin/python"

# Location of the template slurm path used to produce the slurm scripts
slurm_template_file = root_path / "slurm_template.sh"

# Location of where the configuration files will be written to
config_files_path = root_path / "config_files"

# Location to put the slurm scripts
slurm_scripts_path = root_path / "slurm/scripts"

# Location to put the slurm job.out files
slurm_jobs_path = root_path / "slurm/jobs"


def sbatch_config_files(flags: Set[str], job_names: List[str]) -> None:
    """Launch confiuration files using sbatch.

    Parameters
    ----------
    flags : Set[str]
        Set of flags to run runner.main with
    job_names : List[str]
        Sorted names for the slurm jobs
    """

    # Delete old slurm files make the directory if needed
    if slurm_scripts_path.exists():
        for p in slurm_scripts_path.glob("*.sh"):
            p.unlink()
    else:
        slurm_scripts_path.mkdir()

    with open(slurm_template_file, "r", encoding="utf8") as f:
        slurm_lines = f.readlines()

    config_files = sorted(config_files_path.glob("*.json"), key=lambda x: int(x.stem))

    for i, (cfg_pth, name) in enumerate(zip(config_files, job_names)):

        # Replace specific lines with what we need
        for j in range(len(slurm_lines)):
            if "bin/python" in slurm_lines[j]:
                slurm_lines[j] = f"#!{python_path} -u"
            elif "--chdir" in slurm_lines[j]:
                slurm_lines[j] = f"#SBATCH --chdir={root_path}\n"
            elif "--job-name" in slurm_lines[j]:
                slurm_lines[j] = f"#SBATCH --job-name={name}\n"
            elif "--output" in slurm_lines[j]:
                slurm_lines[j] = f"#SBATCH --output={slurm_jobs_path}/job.%A_%a.out\n"
            elif "runner.main" in slurm_lines[j]:
                slurm_lines[j] = f"runner.main(config_file='{cfg_pth.as_posix()}', flags={flags})\n"

        slurm_script_file = slurm_scripts_path / f"{i}.sh"
        with open(slurm_script_file, "w", encoding="utf8") as f:
            f.writelines(slurm_lines)

        result = subprocess.run(
            ["sbatch", slurm_script_file.as_posix()], capture_output=True, check=True
        )
        print(result.stdout)


def create_config_file(experiment_parameters: Dict[str, Union[str, int]], i: int):
    """Create the configuration file for a particular set of hyperparameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]]
        Experiment specifications
    i : int
        Used to give the config file a unique name
    """

    config_file = config_files_path / f"{i}.json"
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(experiment_parameters, f, sort_keys=True, indent=4, separators=(",", ": "))


def main(
    experiment_parameters_lists: Dict[str, List[Union[str, int]]], flags: Set[str], local: bool
) -> None:
    """Create configuration files or run the AL pipeline with a set of hyperparameters.

    Parameters
    ----------
    experiment_parameters_lists : Dict[str, List[Union[str, int]]]
        Lists of sets of experiment parameters
    flags : Set[str]
        Set of flags to run runner.main with
    local : bool
        If True, the experiments should be run locally and config files not produced.
    """

    # Delete old configuration files if running on the cluster and make the directory if needed
    if not local:
        if config_files_path.exists():
            for p in config_files_path.glob("*.json"):
                p.unlink()
        else:
            config_files_path.mkdir()

    job_names = []
    i = 0
    for random_state in experiment_parameters_lists["random_state"]:
        for stop_set_size in experiment_parameters_lists["stop_set_size"]:
            for batch_size in experiment_parameters_lists["batch_size"]:
                for base_learner in experiment_parameters_lists["base_learner"]:
                    for multiclass in experiment_parameters_lists["multiclass"]:
                        for feature_representation in experiment_parameters_lists[
                            "feature_representation"
                        ]:
                            for dataset in experiment_parameters_lists["dataset"]:

                                experiment_parameters = {
                                    "output_root": experiment_parameters_lists["output_root"],
                                    "task": experiment_parameters_lists["task"],
                                    "stop_set_size": stop_set_size,
                                    "batch_size": batch_size,
                                    "feature_representation": feature_representation,
                                    "base_learner": base_learner,
                                    "multiclass": multiclass,
                                    "dataset": dataset,
                                    "random_state": random_state,
                                }
                                experiment_parameters = {
                                    k: str(v) for k, v in experiment_parameters.items()
                                }

                                if local:
                                    runner.main(
                                        experiment_parameters=experiment_parameters, flags=flags
                                    )
                                else:
                                    create_config_file(experiment_parameters, i)

                                job_names.append(dataset)
                                i += 1

        # Averaging across random states only needs to be run once
        if "averaging" in flags:
            break

    if not local:
        sbatch_config_files(flags, job_names)
