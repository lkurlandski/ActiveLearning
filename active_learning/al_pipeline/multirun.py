"""Run multiple experimental configurations using SLURM (with some restrictions).
"""

import argparse
from itertools import product
from pathlib import Path
from pprint import pformat, pprint  # pylint: disable=unused-import
import subprocess
import sys  # pylint: disable=unused-import
from typing import Any, Dict, List, Union

from active_learning.al_pipeline.helpers import Params


# Location of the ActiveLearning directory
root_path = Path("./").resolve()
# Python interpreter path
python_path = root_path / "env/bin/python"
# Directory containing slurm utilities
slurm_path = root_path / "slurm"
# Location to write slurm scripts for sbatch submission
slurm_file = slurm_path / "submission.py"
# Location to put the slurm job.out files
slurm_jobs_path = slurm_path / "jobs"


def main(
    multi_params: Dict[str, Union[str, List[Any]]],
    learn: bool,
    evaluate: bool,
    process: bool,
    stopping: bool,
    graph: bool,
    average: bool,
    cpus_per_task: int,
) -> None:
    """Create and submit individual submissions scripts for different experimental configurations.

    Parameters
    ----------
    multi_params : Params
        A dictionary of lists to represent a series of experimental parameters to run.
    learn : bool
        If true, runs the active learning process.
    evaluate : bool
        If true, runs the evaluation process.
    process : bool
        If true, runs the processing process.
    stopping : bool
        If true, runs the stopping process.
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

    if average and any((learn, evaluate, process, stopping, graph)):
        raise Exception("The averaging program should be run after all runs of pipeline finish.")
    if any(isinstance(multi_params[k], list) for k in ["output_root", "dataset"]):
        raise TypeError("Only one option for 'output_root', 'task', and 'dataset' is permitted.")

    dataset = multi_params["dataset"]
    output_root = multi_params["output_root"]

    slurm_precepts = [
        f"#!{python_path.as_posix()} -u",
        f"#SBATCH --chdir={root_path.as_posix()}",
        f"#SBATCH --output={slurm_jobs_path / f'job.{dataset}.%A.out'}",
        f"#SBATCH --job-name={multi_params['dataset']}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        "#SBATCH --constraint=skylake|broadwell",
        "#SBATCH --partition=long",
        "from active_learning.al_pipeline import run",
        "from active_learning.al_pipeline.helpers import Params",
    ]
    slurm_precepts = [s + "\n" for s in slurm_precepts]

    order = [
        "early_stop_mode",
        "first_batch_mode",
        "batch_size",
        "query_strategy",
        "base_learner",
        "feature_rep",
        "random_state",
    ]

    for combo in product(*[multi_params[k] for k in order]):
        params = Params(
            output_root=output_root,
            early_stop_mode=combo[0],
            first_batch_mode=combo[1],
            batch_size=combo[2],
            query_strategy=combo[3],
            base_learner=combo[4],
            feature_rep=combo[5],
            dataset=dataset,
            random_state=combo[6],
        )

        with open(slurm_file, "w", encoding="utf8") as f:
            f.writelines(slurm_precepts)
            f.write(
                f"run.main(\n"
                f"    {params.construct_str()},\n"
                f"    {learn},\n"
                f"    {evaluate},\n"
                f"    {process},\n"
                f"    {stopping},\n"
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
    parser.add_argument("--stopping", action="store_true", help="Analyze stopping criteria.")
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
        "output_root": "./output",
        "early_stop_mode": ["none"],
        "first_batch_mode": ["random"],
        "batch_size": [0.01],
        "query_strategy": ["closest_to_hyperplane"],
        "base_learner": ["LinearSVC"],
        "feature_rep": ["TfidfVectorizer"],
        "dataset": "20newsgroups-singlelabel",
        "random_state": [0],
    }

    main(
        multi_params_,
        args.learn,
        args.evaluate,
        args.process,
        args.stopping,
        args.graph,
        args.average,
        args.cpus_per_task,
    )
