"""From experiment paramaters, create config files, then slurm scripts, and sbatch them.

TODO
----
-

FIXME
-----
-
"""

from copy import deepcopy
from itertools import product
from pathlib import Path
from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import subprocess
from typing import Any, Dict, List, Set, Union
import warnings


from active_learning import runner


# Location of the ActiveLearning directory
root_path = Path("/home/hpc/kurlanl1/bloodgood/ActiveLearning")

# Interpreter path
python_path = root_path / "env/bin/python"

# Location of where the configuration files will be written to
config_files_path = root_path / "config_files"

# Directory containing slurm utilities
slurm_path = root_path / "slurm"

# Location to write slurm scripts for sbatch submission
slurm_file = slurm_path / "submission.py"

# Location to put the slurm job.out files
slurm_jobs_path = slurm_path / "jobs"

# The number of cpus/cores to allocate for experiments with different datasets
# When performing OVR classification n_classes classifiers are trained and ensembled
# The individual training can be done in parallel, which requires multiple cores
cpus_per_task_by_dataset = {
    "Avila": 12,
    "RCV1_v2": 10,
    "Reuters": 10,
    "WebKB": 4,
    "glue": 1,
    "ag_news": 4,
    "amazon_polarity": 1,
    "emotion": 6,
    "20NewsGroups": 20,
    "Covertype": 7,
    "Iris": 3,
}

default_sbatch_args = {
    "chdir": root_path,
    "output": f"{slurm_jobs_path}/job.%A.out",
    "job-name": "cls",
    "constraint": "skylake|broadwell",
    "partition": "long",
    "cpus-per-task": 1,
}


def create_submission_script(
    experiment_parameters: Dict[str, Any],
    flags: Set[str],
    dataset: str,
) -> None:
    """Create a slurm submission script.

    Parameters
    ----------
    experiment_parameters : Dict[str, Any]
        Experiment parameters for the experiment.
    flags : Set[str]
        Set of flags describing which processes should be run.
    dataset : str
        The dataset which is being used.
    """

    sbatch_args = deepcopy(default_sbatch_args)
    if dataset in cpus_per_task_by_dataset:
        sbatch_args["cpus-per-task"] = cpus_per_task_by_dataset[dataset]
    else:
        warnings.warn(
            f"Allocating one cpu for this task because there is no instruction for {dataset}."
            " This will run one-versus-one classification concurrently instead of in parallel."
        )

    sbatch_args["job-name"] = dataset

    lines = (
        [f"#!{python_path} -u", "\n"]
        + [f"#SBATCH --{k}={v}" for k, v in sbatch_args.items()]
        + [
            "\n",
            "from active_learning import runner",
            "\n" f"runner.main({pformat(experiment_parameters)}, {pformat(flags)})",
        ]
    )

    lines = [l + "\n" for l in lines]
    with open(slurm_file, "w", encoding="utf8") as f:
        f.writelines(lines)


def main(
    experiment_parameters_lists: Dict[str, List[Union[str, int]]],
    flags: Set[str],
    local: bool,
) -> None:
    """Run the AL pipeline with a set of hyperparameters through slurm or locally.

    Parameters
    ----------
    experiment_parameters_lists : Dict[str, List[Union[str, int]]]
        Lists of sets of experiment parameters
    flags : Set[str]
        Set of flags to run runner.main with
    local : bool
        If True, the experiments should be run locally and config files not produced.
    """

    order = [
        "random_state",
        "stop_set_size",
        "batch_size",
        "query_strategy",
        "base_learner",
        "multiclass",
        "feature_representation",
        "dataset",
    ]

    first_random_state = None

    # This usage iterables.product is equivalent to a deeply nested for loop, but requires only
    # one level of indentation
    for (
        random_state,
        stop_set_size,
        batch_size,
        query_strategy,
        base_learner,
        multiclass,
        feature_representation,
        dataset,
    ) in product(*[experiment_parameters_lists[k] for k in order]):

        first_random_state = random_state if first_random_state is None else first_random_state

        experiment_parameters = {
            "output_root": experiment_parameters_lists["output_root"],
            "task": experiment_parameters_lists["task"],
            "stop_set_size": stop_set_size,
            "batch_size": batch_size,
            "query_strategy": query_strategy,
            "feature_representation": feature_representation,
            "base_learner": base_learner,
            "multiclass": multiclass,
            "dataset": dataset,
            "random_state": random_state,
        }
        experiment_parameters = {k: str(v) for k, v in experiment_parameters.items()}

        if local:
            runner.main(experiment_parameters, flags)
        else:
            create_submission_script(experiment_parameters, flags, dataset)
            result = subprocess.run(
                ["sbatch", slurm_file.as_posix()], capture_output=True, check=True
            )
            print(result.stdout.decode().strip())

        # Averaging across random states only needs to be run once
        if "averaging" in flags and first_random_state != random_state:
            break
