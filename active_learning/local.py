"""
Convienient for running processes locally from command line.

TODO
----
-

FIXME
-----
-
"""


experiment_parameters = {
    "output_root": "outputGarbage",
    "task": "cls",
    "stop_set_size": 1,
    "batch_size": 0.07,
    "query_strategy": "uncertainty_batch_sampling",
    "base_learner": "SVC",
    "multiclass": "ovr",
    "feature_representation": "count",
    "dataset": "20NewsGroups",
    "random_state": 0,
}
