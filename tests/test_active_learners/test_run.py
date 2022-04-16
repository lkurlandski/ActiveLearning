"""Test the runner program.
"""

from copy import deepcopy

from active_learning.active_learners import run
from tests import params_iris


def test_run(tmp_path):

    params = deepcopy(params_iris)
    params["output_root"] = tmp_path.as_posix()

    params["random_state"] = 0
    run.main(params, True, True, True, True, False)


def test_average(tmp_path):

    params = deepcopy(params_iris)
    params["output_root"] = tmp_path.as_posix()

    params["random_state"] = 0
    run.main(params, True, True, True, True, False)

    params["random_state"] = 1
    run.main(params, True, True, True, True, False)

    run.main(params, False, False, False, False, True)
