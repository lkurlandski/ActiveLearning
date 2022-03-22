"""Assets that can be used across many different tests.

TODO
----
-

FIXME
-----
-
"""

experiment_parameters_iris = {
    "output_root": None,
    "task": "cls",
    "stop_set_size": 1000,
    "batch_size": 7,
    "base_learner": "SVC",
    "multiclass": "ovr",
    "feature_representation": "preprocessed",
    "dataset": "Iris",
    "random_state": 0,
}
