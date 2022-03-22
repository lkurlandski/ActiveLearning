"""Test the active learning procedure.

TODO
----
-

FIXME
-----
-
"""

from copy import deepcopy

from active_learning import active_learner
from tests import experiment_parameters_iris


def test1(tmp_path):
    experiment_parameters = deepcopy(experiment_parameters_iris)
    experiment_parameters["output_root"] = tmp_path.as_posix()
    active_learner.main(experiment_parameters)
