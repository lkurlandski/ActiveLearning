#!/home/hpc/kurlanl1/bloodgood/ActiveLearning/env/bin/python -u

# SBATCH --chdir=/home/hpc/kurlanl1/bloodgood/ActiveLearning
# SBATCH --output=/home/hpc/kurlanl1/bloodgood/ActiveLearning/slurm/jobs/job.20NewsGroups.%A.out
# SBATCH --constraint=skylake|broadwell
# SBATCH --job-name=20NewsGroups
# SBATCH --partition=long
# SBATCH --cpus-per-task=1

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
from active_learning.active_learners import stopping


def main(
    params: Dict[str, Union[str, int]],
    learn_: bool,
    evaluate_: bool,
    process_: bool,
    graph_: bool,
    stopping_: bool,
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
    stopping : bool
        If true, runs the stopping process.
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
    print(f"\tevaluate: {evaluate_}", flush=True)
    print(f"\tprocess: {process_}", flush=True)
    print(f"\tgraph: {graph_}", flush=True)
    print(f"\tstopping: {stopping_}", flush=True)
    print(f"\taverage: {average_}", flush=True)
    print(f"\nExperimental parameters:\n{pformat(params)}")

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

    if stopping_:
        stopping.main(params)

    if average_:
        average.main(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learn", action="store_true", help="Perform active learning.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the learned models.")
    parser.add_argument("--process", action="store_true", help="Process evaluations into tables.")
    parser.add_argument("--graph", action="store_true", help="Create basic graphs for statistics.")
    parser.add_argument("--stopping", action="store_true", help="Analyze stopping criteria.")
    parser.add_argument("--average", action="store_true", help="Average across multiple runs.")

    args = parser.parse_args()

    params_ = {
        "output_root": "outputs/test",
        "task": "cls",
        "early_stop_mode": "exponential",
        "first_batch_mode": "random",
        "batch_size": 1,
        "query_strategy": "uncertainty_sampling",
        "base_learner": "SVC",
        "feature_representation": "preprocessed",
        "dataset": "Iris",
        "random_state": 0,
    }

    main(
        params_,
        args.learn,
        args.evaluate,
        args.process,
        args.graph,
        args.stopping,
        args.average,
    )
