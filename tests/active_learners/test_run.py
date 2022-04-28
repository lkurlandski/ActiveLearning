"""Test the runner program.
"""

from copy import deepcopy

from active_learning.active_learners import run
from active_learning.active_learners.helpers import Params


params = Params(
    output_root="outputs/garbage",
    early_stop_mode="none",
    first_batch_mode="random",
    batch_size=7,
    query_strategy="uncertainty_sampling",
    base_learner="SVC",
    feature_representation="preprocessed",
    dataset="Iris",
    random_state=0,
)


def test_pipeline(tmp_path):

    params.output_root = tmp_path
    run.main(params, True, True, True, True, True, False)

    params.random_state = 1
    run.main(params, True, True, True, True, True, False)

    run.main(params, False, False, False, False, False, True)
