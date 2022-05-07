"""Experiment with different stopping crteria after active learning commences.
"""

import datetime
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
import time
from typing import List

import joblib
import numpy as np
import pandas as pd

from active_learning import stat_helper
from active_learning.stopping_criteria.base import StoppingCriteria
from active_learning.stopping_criteria import StabilizingPredictions
from active_learning.al_pipeline.helpers import (
    IndividualOutputDataContainer,
    OutputHelper,
    Params,
    Pool,
)


def run_stopping_criteria(
    criteria: List[StoppingCriteria],
    container: IndividualOutputDataContainer,
) -> None:
    """Run the various stopping criteria and determine when they should stop.

    Parameters
    ----------
    criteria : List[StoppingCriteria]
        A list of stopping criteria to experiment with.
    container : IndividualOutputDataContainer
        A container containing the experimental outputs to run the stopping criteria on.
    """
    print("-" * 80, flush=True)
    start = time.time()
    print("0:00:00 -- Starting Stopping", flush=True)

    unlabeled_pool = Pool(
        X_path=container.X_unlabeled_pool_file, y_path=container.y_unlabeled_pool_file
    ).load()
    initial_unlabeled_pool = Pool(X=unlabeled_pool.X.copy(), y=unlabeled_pool.y.copy)

    results = pd.DataFrame(columns=["iteration", "training_data", "criteria"])
    training_data = 0

    n_iterations = sum(1 for _ in container.model_path.iterdir())
    for i in range(n_iterations):

        model = joblib.load(container.model_path / f"{i}.joblib")

        idx = np.load(container.batch_path / f"{i}.npy")
        unlabeled_pool.X = stat_helper.remove_ids_from_array(unlabeled_pool.X, idx)
        unlabeled_pool.y = stat_helper.remove_ids_from_array(unlabeled_pool.y, idx)
        training_data += len(idx)

        initial_unlabeled_pool_preds = model.predict(initial_unlabeled_pool.X)

        for c in criteria:
            if c.has_stopped:
                continue

            if isinstance(c, StabilizingPredictions):
                c.update_from_preds(initial_unlabeled_pool_preds=initial_unlabeled_pool_preds)

            if c.has_stopped:
                df = pd.DataFrame(
                    {
                        "iteration": [i],
                        "training_data": [training_data],
                        "criteria": [str(c)],
                    }
                )
                results = results.append(df)

    never_stopped_criteria = [c for c in criteria if not c.has_stopped]
    df = pd.DataFrame(
        {
            "iteration": [np.NaN] * len(never_stopped_criteria),
            "training_data": [np.NaN] * len(never_stopped_criteria),
            "criteria": [str(c) for c in never_stopped_criteria],
        }
    )
    results = results.append(df)
    results.to_csv(container.stopping_results_file)

    diff = datetime.timedelta(seconds=(round(time.time() - start)))
    print(f"{diff} -- Ending Stopping", flush=True)
    print("-" * 80, flush=True)


def main(params: Params) -> None:
    """Run the stopping criteria from a set of experimental parameters.

    Parameters
    ----------
    params : Params
        Experimental paramters.
    """
    criteria = []

    for windows in (1, 2, 3, 4):
        for threshold in (0.96, 0.97, 0.98, 0.99):
            for stop_set_size in (0.2, 0.4, 0.6, 0.8):
                criteria.append(
                    StabilizingPredictions(
                        windows=windows,
                        threshold=threshold,
                        stop_set_size=stop_set_size,
                    )
                )

    oh = OutputHelper(params)
    run_stopping_criteria(criteria, oh.container)
