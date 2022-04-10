#!/home/hpc/kurlanl1/bloodgood/ActiveLearning/env/bin/python -u

# SBATCH --chdir=/home/hpc/kurlanl1/bloodgood/ActiveLearning
# SBATCH --output=/home/hpc/kurlanl1/bloodgood/ActiveLearning/slurm/jobs/job.%A.out
# SBATCH --job-name=20NewsGroups
# SBATCH --constraint=skylake|broadwell
# SBATCH --partition=long
# SBATCH --cpus-per-task=20

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

    params__ = {
        "output_root": "outputs/performance",
        "task": "cls",
        "stop_set_size": 0.1,
        "batch_size": 0.4,
        "query_strategy": "uncertainty_sampling",
        "base_learner": "SVC",
        "multiclass": "ovr",
        "feature_representation": "preprocessed",
        "dataset": "Iris",
        "random_state": 0,
    }

    learn__ = args.learn or False
    evaluate__ = args.evaluate or False
    process__ = args.process or False
    graph__ = args.graph or False
    average__ = args.average or False

    main(
        params__,
        learn__,
        evaluate__,
        process__,
        graph__,
        average__,
    )
