"""Test the runner program.
"""

from copy import deepcopy

from active_learning import runner
from active_learning import utils
from tests import experiment_parameters_iris


def test1(tmp_path):

    experiment_parameters = deepcopy(experiment_parameters_iris)
    experiment_parameters["output_root"] = tmp_path.as_posix()

    experiment_parameters["random_state"] = 0
    runner.main(
        experiment_parameters=experiment_parameters,
        flags={"active_learning", "processor", "graphing"},
    )

    # Run a different random state then average them
    experiment_parameters["random_state"] = 1
    runner.main(
        experiment_parameters=experiment_parameters,
        flags={"active_learning", "processor", "graphing", "averaging"},
    )

    print("\n".join(list(utils.tree(tmp_path))))
