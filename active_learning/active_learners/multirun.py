"""Run multiple experimental configurations using SLURM (with some restrictions).
"""

import argparse
from itertools import product
from pathlib import Path
from pprint import pformat
import subprocess
from typing import Any, Dict, List


# Location of the ActiveLearning directory
root_path = Path("/path/to/ActiveLearning")
# Python interpreter path
python_path = root_path / "env/bin/python"
# Directory containing slurm utilities
slurm_path = root_path / "slurm"
# Location to write slurm scripts for sbatch submission
slurm_file = slurm_path / "submission.py"
# Location to put the slurm job.out files
slurm_jobs_path = slurm_path / "jobs"


def main(
    multi_params: Dict[str, List[Any]],
    learn: bool,
    evaluate: bool,
    process: bool,
    graph: bool,
    average: bool,
    cpus_per_task: int,
) -> None:
    """Create and submit individual submissions scripts for different experimental configurations.

    Parameters
    ----------
    multi_params : Dict[str, List[Any]]
        A dictionary of lists to represent a series of experimental parameters to run.
    learn : bool
        If true, runs the active learning process.
    evaluate : bool
        If true, runs the evaluation process.
    process : bool
        If true, runs the processing process.
    graph : bool
        If true, runs the graphing process.
    average : bool
        If true, runs the averaging process.
    cpus_per_task : int
        Number of cpus to allocate to SLURM using sbatch.

    Raises
    ------
    Exception
        If the both averaging program and any other programs are requested.
    TypeError
        If the multiple output_roots, tasks or datasets are passed as parameters.
    """

    print(f"multirun -- params:\n{pformat(multi_params)}")

    if average and any(learn, evaluate, process, graph):
        raise Exception("The averaging program should be run after all runs of pipeline finish.")
    if any(isinstance(multi_params[k], list) for k in ["output_root", "task", "dataset"]):
        raise TypeError("Only one option for 'output_root', 'task', and 'dataset' is permitted.")

    dataset = multi_params["dataset"]
    output_root = multi_params["output_root"]
    task = multi_params["task"]

    slurm_precepts = [
        f"#!{python_path.as_posix()} -u",
        f"#SBATCH --chdir={root_path.as_posix()}",
        f"#SBATCH --output={slurm_jobs_path / f'job.{dataset}.%A.out'}",
        f"#SBATCH --job-name={multi_params['dataset']}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        "#SBATCH --constraint=skylake|broadwell",
        "#SBATCH --partition=long",
        "from active_learning.active_learners import run",
    ]
    slurm_precepts = [s + "\n" for s in slurm_precepts]

    order = [
        "random_state",
        "stop_set_size",
        "batch_size",
        "query_strategy",
        "base_learner",
        "multiclass",
        "feature_representation",
    ]

    for combo in product(*[multi_params[k] for k in order]):

        random_state = combo[0]
        stop_set_size = combo[1]
        batch_size = combo[2]
        query_strategy = combo[3]
        base_learner = combo[4]
        multiclass = combo[5]
        feature_representation = combo[6]

        params = {
            "output_root": output_root,
            "task": task,
            "stop_set_size": stop_set_size,
            "batch_size": batch_size,
            "query_strategy": query_strategy,
            "feature_representation": feature_representation,
            "base_learner": base_learner,
            "multiclass": multiclass,
            "dataset": dataset,
            "random_state": random_state,
        }
        params = {k: str(v) for k, v in params.items()}

        with open(slurm_file, "w", encoding="utf8") as f:
            f.writelines(slurm_precepts)
            f.write(
                f"run.main(\n"
                f"    {pformat(params, indent=8)},\n"
                f"    {learn},\n"
                f"    {evaluate},\n"
                f"    {process},\n"
                f"    {graph},\n"
                f"    {average},\n"
                ")"
                "\n"
            )

        result = subprocess.run(["sbatch", slurm_file.as_posix()], capture_output=True, check=True)
        print(result.stdout.decode().strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learn", action="store_true", help="Perform active learning.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the learned models.")
    parser.add_argument("--process", action="store_true", help="Process evaluations into tables.")
    parser.add_argument("--graph", action="store_true", help="Create basic graphs for statistics.")
    parser.add_argument("--average", action="store_true", help="Average across multiple runs.")
    parser.add_argument(
        "--cpus_per_task",
        action="store",
        default=1,
        help="Number of cpus required per task minimum number of logical processors (threads).",
    )

    args = parser.parse_args()

    multi_params_ = {
        "output_root": "outputs/output",
        "task": "cls",
        "stop_set_size": [0.1],
        "batch_size": [0.10],
        "query_strategy": ["uncertainty_sampling"],
        "base_learner": ["SVC"],
        "multiclass": ["ovr"],
        "feature_representation": ["tfidf"],
        "dataset": "20NewsGroups",
        "random_state": [0],
    }

    main(
        multi_params_,
        args.learn,
        args.evaluate,
        args.process,
        args.graph,
        args.average,
        args.cpus_per_task,
    )
