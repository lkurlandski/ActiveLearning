"""Test the runner program.
"""

from active_learning.al_pipeline import run
from active_learning.al_pipeline.helpers import Params


def test_pipeline_iris(tmp_path):
    params = Params(
        output_root=tmp_path,
        early_stop_mode="none",
        first_batch_mode="random",
        batch_size=7,
        query_strategy="uncertainty_sampling",
        base_learner="SVC",
        feature_rep="none",
        dataset="iris",
        random_state=0,
    )

    run.main(params, True, True, True, True, True, False)


def test_pipeline_20newsgroups(tmp_path):
    params = Params(
        output_root=tmp_path,
        early_stop_mode="none",
        first_batch_mode="random",
        batch_size=0.7,
        query_strategy="closest_to_hyperplane",
        base_learner="SGDClassifier",
        feature_rep="CountVectorizer",
        dataset="20newsgroups-singlelabel",
        random_state=0,
    )

    run.main(params, True, True, True, True, True, False)


def test_pipeline_reuters(tmp_path):
    params = Params(
        output_root=tmp_path,
        early_stop_mode="none",
        first_batch_mode="random",
        batch_size=0.7,
        query_strategy="closest_to_hyperplane",
        base_learner="LinearSVC",
        feature_rep="CountVectorizer",
        dataset="reuters",
        random_state=0,
    )

    run.main(params, True, True, True, True, True, False)


def test_full_pipeline(tmp_path):
    params = Params(
        output_root=tmp_path,
        early_stop_mode="none",
        first_batch_mode="random",
        batch_size=7,
        query_strategy="uncertainty_sampling",
        base_learner="SVC",
        feature_rep="none",
        dataset="iris",
        random_state=0,
    )

    run.main(params, True, True, True, True, True, False)

    params.random_state = 1
    run.main(params, True, True, True, True, True, False)

    run.main(params, False, False, False, False, False, True)
