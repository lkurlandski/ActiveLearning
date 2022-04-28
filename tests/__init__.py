"""Assets that can be used across many different tests.
"""

params_iris = {
    "output_root": None,
    "task": "cls",
    "early_stop_mode": "none",
    "first_batch_mode": "random",
    "batch_size": 7,
    "query_strategy": "uncertainty_batch_sampling",
    "base_learner": "SVC",
    "multiclass": "ovr",
    "feature_representation": "preprocessed",
    "dataset": "Iris",
    "random_state": 0,
}
