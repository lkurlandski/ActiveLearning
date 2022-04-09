"""Main program to execute processed for a specific set of experiment parameters.

TODO
----
- Decide upon a better naming convention for the flags.

FIXME
-----
-
"""

import argparse
import json
from pathlib import Path
from pprint import pformat
from typing import Dict, Set, Union
import warnings

from sklearn.exceptions import ConvergenceWarning

from active_learning import active_learner
from active_learning import averager
from active_learning import graphing
from active_learning import local
from active_learning import processor


all_flags = {"active_learning", "processor", "graphing", "averaging"}
default_flags = {"active_learning", "processor", "graphing"}


def main(
    experiment_parameters: Dict[str, Union[str, int]],
    flags: Set[str] = None,
):
    """Run the AL pipeline from a set of experiment parameters.

    Parameters
    ----------
    experiment_parameters : Dict[str, Union[str, int]], optional
        A single set of hyperparmaters and for the active learning experiment.
    flags : Set[str], optional
        Set of flags to control which parts of the AL pipeline are run, by default None.

    Raises
    ------
    ValueError
        If flags contains an unrecognized flag.
    """

    flags = default_flags if flags is None else flags
    print(f"runner.main -- flags:\n{pformat(flags)}")

    if not flags <= all_flags:
        raise ValueError(
            f"Unexpected flags passed as argument: {flags}." f"\nValid flags are: {all_flags}"
        )

    print(f"runner.main -- experiment_parameters:\n{pformat(experiment_parameters)}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if "active_learning" in flags:
            active_learner.main(experiment_parameters)

    if "processor" in flags:
        processor.main(experiment_parameters)

    if "graphing" in flags:
        graphing.main(experiment_parameters)

    if "averaging" in flags:
        averager.main(experiment_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--averager",
        help="Run the averager program instead of pipeline",
        action="store_true",
    )
    args = parser.parse_args()

    main(
        local.experiment_parameters,
        default_flags if not args.averager else "averaging",
    )
