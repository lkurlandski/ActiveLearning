#!/home/hpc/elphicb1/ActiveLearning/ActiveLearning/env2/bin/python -u

#SBATCH --chdir=/home/hpc/elphicb1/ActiveLearning/ActiveLearning
#SBATCH --output=/home/hpc/elphicb1/ActiveLearning/ActiveLearning/slurm/jobs/job.%A.out
#SBATCH --constraint=skylake|broadwell
#SBATCH --job-name=name
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1

"""Interface to the features of this module that can be used with python or sbatch.

To interface properly with SLURM, the top of the file should contain the SLURM instructions.
"""

import argparse
import datetime
from pprint import pformat
import time
import warnings

from sklearn.exceptions import ConvergenceWarning

from active_learning.active_learners import average
from active_learning.active_learners import evaluate
from active_learning.active_learners import graph
from active_learning.active_learners import learn
from active_learning.active_learners import process
from active_learning.active_learners import stopping
from active_learning.active_learners.helpers import Params


def main(
    params: Params,
    learn_: bool,
    evaluate_: bool,
    process_: bool,
    stopping_: bool,
    graph_: bool,
    average_: bool,
):
    """Create and submit individual submissions scripts for different experimental configurations.

    Parameters
    ----------
    params : Params
        A single set of experimental parameters.
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

    Raises
    ------
    Exception
        If the both averaging program and any other programs are requested.
    """
    print("-" * 80, flush=True)
    start = time.time()
    print("0:00:00 -- Starting Active Learning Pipeline", flush=True)

    print("Tasks to run: ", flush=True)
    print(f"\tlearn: {learn_}", flush=True)
    print(f"\tevaluate: {evaluate_}", flush=True)
    print(f"\tprocess: {process_}", flush=True)
    print(f"\tstopping: {stopping_}", flush=True)
    print(f"\tgraph: {graph_}", flush=True)
    print(f"\taverage: {average_}", flush=True)
    print(f"\nExperimental parameters:\n{pformat(params)}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if learn_:
            learn.main(params)

    if evaluate_:
        evaluate.main(params)

    if process_:
        process.main(params)

    if stopping_:
        stopping.main(params)

    if graph_:
        graph.main(params)

    if average_:
        average.main(params)

    diff = datetime.timedelta(seconds=(round(time.time() - start)))
    print(f"{diff} -- Ending Active Learning Pipeline", flush=True)
    print("-" * 80, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learn", action="store_true", help="Perform active learning.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the learned models.")
    parser.add_argument("--process", action="store_true", help="Process evaluations into tables.")
    parser.add_argument("--graph", action="store_true", help="Create basic graphs for statistics.")
    parser.add_argument("--stopping", action="store_true", help="Analyze stopping criteria.")
    parser.add_argument("--average", action="store_true", help="Average across multiple runs.")

    args = parser.parse_args()

    params_ = Params(
        output_root="/home/hpc/elphicb1/ActiveLearning/ActiveLearning/new_output/",
        early_stop_mode="none",
        first_batch_mode="random",
        batch_size=.02,
        query_strategy="uncertainty_sampling",
        base_learner="SVC",
        feature_rep="fasttext",
        dataset="rotten_tomatoes",
        random_state=0,
    )

    main(
        params_,
        args.learn,
        args.evaluate,
        args.process,
        args.graph,
        args.stopping,
        args.average,
    )
