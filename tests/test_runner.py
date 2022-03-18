from active_learning import runner
from active_learning import utils


def test1(tmp_path):

    experiment_parameters = {
        "output_root": tmp_path.as_posix(),
        "task": "cls",
        "stop_set_size": 1000,
        "batch_size": 7,
        "base_learner": "SVC",
        "multiclass": "ovr",
        "feature_representation": "preprocessed",
        "dataset": "Iris",
        "random_state": 0,
    }

    experiment_parameters["random_state"] = 0
    runner.main(
        experiment_parameters=experiment_parameters,
        flags={"active_learning", "processor", "graphing"},
    )

    experiment_parameters["random_state"] = 1
    runner.main(
        experiment_parameters=experiment_parameters,
        flags={"active_learning", "processor", "graphing", "averaging"},
    )

    print("\n".join(list(utils.tree(tmp_path))))
    assert 0
