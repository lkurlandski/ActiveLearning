#!/path/to/ActiveLearning/env/bin/python -u

#SBATCH --chdir=/path/to/ActiveLearning
#SBATCH --output=/path/to/ActiveLearning/slurm/jobs/job.name.%A.out
#SBATCH --constraint=skylake|broadwell
#SBATCH --job-name=name
#SBATCH --partition=long
#SBATCH --cpus-per-task=1

"""Interface to the features of this module that can be used with python or sbatch.

To interface properly with SLURM, the top of the file should contain the SLURM instructions.
"""

import argparse
from pprint import pformat
from typing import Dict, Union
import warnings

from sklearn.exceptions import ConvergenceWarning

from active_learning.active_learners import average
from active_learning.active_learners import evaluate
from active_learning.active_learners import graph
from active_learning.active_learners import learn
from active_learning.active_learners import process


def main(
    params: Dict[str, Union[str, int]],
    learn_: bool,
    evaluate_: bool,
    process_: bool,
    graph_: bool,
    average_: bool,
):
    """Create and submit individual submissions scripts for different experimental configurations.

    Parameters
    ----------
    params : Dict[str, List[Any]]
        A single set of experimental parameters.
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
    """
    print("Tasks to run: ", flush=True)
    print(f"\tlearn: {learn_}", flush=True)
    print(f"\t:evaluate {evaluate_}", flush=True)
    print(f"\t:process {process_}", flush=True)
    print(f"\t:graph {graph_}", flush=True)
    print(f"Experimental parameters:\n{pformat(params)}")

    if average_ and any((learn_, evaluate_, process_, graph_)):
        raise Exception("The averaging program should be run after all runs of pipeline finish.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if learn_:
            learn.main(params)

    if evaluate_:
        evaluate.main(params)

    if process_:
        process.main(params)

    if graph_:
        graph.main(params)

    if average_:
        average.main(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learn", action="store_true", help="Perform active learning.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the learned models.")
    parser.add_argument("--process", action="store_true", help="Process evaluations into tables.")
    parser.add_argument("--graph", action="store_true", help="Create basic graphs for statistics.")
    parser.add_argument("--average", action="store_true", help="Average across multiple runs.")

    args = parser.parse_args()

    params_ = {
        "output_root": "outputs/test",
        "task": "cls",
        "stop_set_size": 0.1,
        "batch_size": 0.01,
        "query_strategy": "uncertainty_sampling",
        "base_learner": "SVC",
        "multiclass": "ovr",
        "feature_representation": "tfidf",
        "dataset": "20NewsGroups",
        "random_state": 0,
    }

    main(
        params_,
        args.learn,
        args.evaluate,
        args.process,
        args.graph,
        args.average,
    )
