"""Collection of valid values for various aspects of the AL system to support vertification.
"""

valid_early_stop_modes = {"exponential", "finish", "none"}


valid_first_batch_modes = {"random", "k_per_class"}


valid_query_strategies = {
    "entropy_sampling",
    "margin_sampling",
    "uncertainty_batch_sampling",
    "uncertainty_sampling",
}
