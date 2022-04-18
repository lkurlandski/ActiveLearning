"""
"""

import datetime
from pprint import pprint
import time
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from active_learning import stat_helper
from active_learning.active_learners import output_helper
from active_learning.active_learners import pool
from active_learning.stopping_criteria.base import StoppingCriteria
from active_learning.stopping_criteria.contradictory_information import (
    ContradictoryInformation,
)
from active_learning.stopping_criteria.stabilizing_predictions import (
    StabilizingPredictions,
)


def test_stopping_criteria(
    criteria: List[StoppingCriteria],
    container: output_helper.IndividualOutputDataContainer,
) -> None:

    start = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(time.time() - start))))} -- Starting Stopping")

    unlabeled_pool = pool.Pool(
        X_path=container.X_unlabeled_pool_file, y_path=container.y_unlabeled_pool_file
    ).load()
    initial_unlabeled_pool = pool.Pool(X=unlabeled_pool.X.copy(), y=unlabeled_pool.y.copy)

    results = pd.DataFrame(columns=["iteration", "training_data"])
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
            elif isinstance(c, ContradictoryInformation):
                pass

            if c.has_stopped:
                df = pd.DataFrame(
                    {
                        "criteria": [str(c)],
                        "iteration": [i],
                        "training_data": [training_data],
                    }
                )
                results = results.append(df)

    never_stopped_criteria = [c for c in criteria if not c.has_stopped]
    df = pd.DataFrame(
        {
            "criteria": [str(c) for c in never_stopped_criteria],
            "iteration": [np.NaN] * len(never_stopped_criteria),
            "training_data": [np.NaN] * len(never_stopped_criteria),
        }
    )
    results.to_csv(container.stopping_results_file)

    end = time.time()
    print(f"{str(datetime.timedelta(seconds=(round(end - start))))} -- Ending Stopping")


def main(params: Dict[str, Union[str, int]]):

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

    oh = output_helper.OutputHelper(params)
    test_stopping_criteria(criteria, oh.container)
