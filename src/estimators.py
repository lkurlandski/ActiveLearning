"""Acquire a scikit-learn estimator from a string code.

This module is heavily reliant upon scikit-learn. Users are advised to refer to the following docs:
    1.12. Multiclass and multioutput algorithms
        https://scikit-learn.org/stable/modules/multiclass.html
"""
from dataclasses import dataclass, field
from enum import Enum
from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Set, Union

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC

import utils


class MCOpts(Enum):
    """The options for multiclass classification supported by the system."""

    ovo: str = "ovo"
    ovr: str = "ovr"
    occ: str = "occ"
    crammer_singer: str = "crammer_singer"
    base: str = "base"


# Mapper to handle the two different names for classification types in sklearn
synonyms = {
    MCOpts.ovo: "one_versus_one",
    MCOpts.ovr: "one_versus_rest",
}


@dataclass
class MultiClassEstimator:
    """Contains the optional multiclass estimator the estimator-creation chain.

    Parameters
    ----------
    wrapper : Callable[..., sklearn.base.BaseEstimator]
        A callable which resutrn a scikit-learn multiclass wrapper
    kwargs : Dict[str, Any]
        Keyword arguments to pass to the wrapper,
            defaults to a dict instructing for maximum parallelization
    """

    wrapper: Callable[..., sklearn.base.BaseEstimator]
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"n_jobs": -1})


def get_multiclass_estimator_map() -> Dict[str, MultiClassEstimator]:
    """Return the mapping of keyword arguments to MultiClassEstimator instances.

    Returns
    -------
    Dict[str, MultiClassEstimator]
        Mapping of keyword arguments to BaseLearner instances
    """

    mapper = {
        MCOpts.occ: MultiClassEstimator(OutputCodeClassifier, {"n_jobs": -1, "random_state": 0}),
        MCOpts.ovo: MultiClassEstimator(OneVsOneClassifier),
        MCOpts.ovr: MultiClassEstimator(OneVsRestClassifier),
    }

    return mapper


@dataclass
class BaseLearner:
    """Contains the base learner as part of the estimator-creation chain.

    Parameters
    ----------
    learner : Callable[..., sklearn.base.BaseEstimator]
        A callable which returns a scikit-learn base learner
    default_multiclass_code : MCOpts
        A string which contains the built-in multiclass support mode of the learner
    kwargs : Dict[str, Any]
        Keyword arguments to pass to the learner, defaults to an empty dict
    multiclass_options: Set[str]
        Set of valid options for the learner 'multi_class' argument, if it has one,
            defaults to an empty set
    """

    learner: Callable[..., sklearn.base.BaseEstimator]
    default_multiclass_code: MCOpts
    kwargs: Dict[str, Any] = field(default_factory=dict)
    multiclass_options: Set[str] = field(default_factory=set)


def get_base_learner_map(probabalistic_required: bool) -> Dict[str, BaseLearner]:
    """Return the mapping of keyword arguments to BaseLearner instances.

    Parameters
    ----------
    probabalistic_required : bool
        Whether or not the system demands probabalistic models, ie, models that implement a
            predict_proba method.

    Returns
    -------
    Dict[str, BaseLearner]
        Mapping of keyword arguments to BaseLearner instances
    """

    # TODO: Investigate MLPClassifier with early_stopping=True destroyed performance on Iris dataset
    # Add learners alphabeically, by the name of their parent package
    mapper = {
        "RandomForestClassifier": BaseLearner(RandomForestClassifier, MCOpts.base),
        "MLPClassifier": BaseLearner(MLPClassifier, MCOpts.base, {"early_stopping": False}),
        "LinearSVC": BaseLearner(
            LinearSVC, MCOpts.ovr, multiclass_options={MCOpts.ovr, MCOpts.crammer_singer}
        ),
        "SVC": BaseLearner(SVC, MCOpts.ovo, {"probability": probabalistic_required}),
        "NuSVC": BaseLearner(
            NuSVC, MCOpts.ovo, {"nu": 0.25, "probability": probabalistic_required}
        ),
    }

    return mapper


def extract_estimator(base_learner: BaseLearner) -> sklearn.base.BaseEstimator:
    """Extract the base learner from the BaseLearner type.

    Parameters
    ----------
    base_learner : BaseLearner
        BaseLearner container from which to extract the base learner estimator from

    Returns
    -------
    sklearn.base.BaseEstimator
        scikit learn base learner
    """

    return base_learner.learner(**base_learner.kwargs)


def wrap_estimator_for_multiclass(
    base_learner: BaseLearner, multiclass_estimator: MultiClassEstimator
) -> Union[OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier]:
    """Instatiated scikit learn base learner then wrap in a multiclass meta estimator.

    Parameters
    ----------
    base_learner : BaseLearner
        BaseLearner object, which will have its base_learner instatiated then wrapped
    multiclass_estimator : MultiClassEstimator
        MultiClassEstimator to instatiate with the base_learner

    Returns
    -------
    Union[OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier]
        A multiclass wrapper around a base learner
    """

    estimator = extract_estimator(base_learner)
    estimator = multiclass_estimator.wrapper(estimator, **multiclass_estimator.kwargs)
    return estimator


def change_estimator_parameters_for_multiclass(
    base_learner: BaseLearner, multiclass_estimator: MultiClassEstimator, multiclass_code: MCOpts
) -> Union[
    sklearn.base.BaseEstimator, OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
]:
    """Alter the multi_class parameter scitkit learn base learner with valid value, or use wrapper.

    Parameters
    ----------
    base_learner : BaseLearner
        BaseLearner object, which will have its base_learner instatiated then possibly wrapped
    multiclass_estimator : MultiClassEstimator
        MultiClassEstimator to possibly instatiate with the base_learner
    multiclass_code : MCOpts
        Code to attempt to use as the base learner's multi_class argument

    Returns
    -------
    Union[sklearn.base.BaseEstimator, OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier]
        Either the scikit learn base learner instatiated with a valid option for its multi_class
            parameter or a multiclass wrapper if requested option is invalid.
    """

    # Use the short string for this multiclass method
    if multiclass_code in base_learner.multiclass_options:
        base_learner.kwargs["multi_class"] = multiclass_code.value
        return extract_estimator(base_learner)
    # Use the alternative string for this multiclass method
    if (
        multiclass_code in synonyms and synonyms[multiclass_code] in base_learner.multiclass_options
    ):
        base_learner.kwargs["multi_class"] = synonyms[multiclass_code]
        return extract_estimator(base_learner)
    # Requested protocal was invalid
    return wrap_estimator_for_multiclass(base_learner, multiclass_estimator)


def is_multiclass_trivial(
    n_targets: int, base_learner: BaseLearner, multiclass_code: MCOpts
) -> bool:
    """Determine if multiclass classification protocols are unnessecary.

    This can occur if:
        - binary classification problem
        - learner handles multiclass naturally
        - learner's default protocol is requested

    Parameters
    ----------
    n_targets : int
        Number of classification targets
    base_learner : BaseLearner
        BaseLearner object
    multiclass_code : MCOpts
        Protocol for which mode of multiclass classification should be deployed

    Returns
    -------
    bool
        Whether or not the multiclass problem needs to be handled
    """

    multiclass_trivial = (
        n_targets == 2
        or base_learner.default_multiclass_code == MCOpts.base
        or base_learner.default_multiclass_code == multiclass_code
    )

    return multiclass_trivial


def get_estimator(
    base_learner_code: str, multiclass_code: str, n_targets: int, probabalistic_required: bool
) -> Union[
    sklearn.base.BaseEstimator,
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
    CalibratedClassifierCV,
]:
    """Get a scikit learn estimator given a desired base learner and a multiclass protocol.

    Parameters
    ----------
    base_learner_code : str
        Code to refer to the base learner to use
    multiclass_code : str
        Code to refer to the multiclass protocol. One of {"ovo", "ovr", "occ", "crammer_singer"}
    n_targets : int
        Number of classification targets
    probabalistic_required : bool
        Whether a probablistic model is required or not, ie, if the model needs to have a
            predict_proba method

    Returns
    -------
    Union[
        sklearn.base.BaseEstimator,
        OneVsOneClassifier,
        OneVsRestClassifier,
        OutputCodeClassifier,
        CalibratedClassifierCV
    ]
        A scikit learn estimator that can be used for multiclass classification

    Raises
    ------
    ValueError
        If the base_learner_code is not recognized
    ValueError
        If crammer_singer multiclass classification is requested for learner other than LinearSVC
    ValueError
        If occ multiclass classification is requested for probablistic models
    """

    # Attain data about the base learner requested
    learner_map = get_base_learner_map(probabalistic_required)
    if base_learner_code not in learner_map:
        raise ValueError(
            f"{base_learner_code} not recognized as a valid base learner."
            f"Valid options are: {set(learner_map.keys())}"
        )
    base_learner = learner_map[base_learner_code]

    # Attain data about the multiclass mode requested
    multiclass_code = MCOpts(multiclass_code)
    multiclass_estimator_map = get_multiclass_estimator_map()
    multiclass_estimator = multiclass_estimator_map[multiclass_code]

    multiclass_trivial = is_multiclass_trivial(n_targets, base_learner, multiclass_code)

    # Check for errors if bad multiclass parameters are passed in
    if not multiclass_trivial:
        if multiclass_code == MCOpts.crammer_singer and not isinstance(
            base_learner.learner, LinearSVC
        ):
            raise ValueError("crammer_singer multiclass mode is only implemented for LinearSVC.")

        if multiclass_code == MCOpts.occ and probabalistic_required:
            raise ValueError(f"{OutputCodeClassifier} does not support probablistic outputs.")

    if multiclass_trivial:
        estimator = extract_estimator(base_learner)
    elif not utils.check_callable_has_parameter(base_learner.learner, "multi_class"):
        estimator = wrap_estimator_for_multiclass(base_learner, multiclass_estimator)
    else:
        estimator = change_estimator_parameters_for_multiclass(
            base_learner, multiclass_estimator, multiclass_code
        )

    # If the base learner not support probabalistic outputs, wrap with probablistic estimator
    if probabalistic_required and not hasattr(estimator, "predict_proba"):
        # FIXME: this will crash if less than 2 examples from any class exist...
        estimator = CalibratedClassifierCV(estimator, cv=2, n_jobs=-1)

    return estimator


def test():
    """Test."""
    import warnings
    from sklearn.datasets import make_classification
    from sklearn.exceptions import ConvergenceWarning

    print("\nTests:\n_____\n\n")
    learner_map = get_base_learner_map(True)
    print(f"learner_map:-----------\n{pformat(learner_map)}\n\n")
    multiclass_estimator_map = get_multiclass_estimator_map()
    print(f"multiclass_estimator_map:-------------\n{pformat(multiclass_estimator_map)}\n\n")

    print("get_estimator:\n-----------------------\n")
    learner_codes = list(learner_map.keys())
    multiclass_codes = list(multiclass_estimator_map.keys())

    X, y1 = make_classification(n_classes=2)
    X, y2 = make_classification(n_classes=4, n_informative=4)

    for y in (y1, y2):
        print(len(np.unique(y)))
        for l in learner_codes:
            print(f"\t{l}")
            for m in multiclass_codes:
                print(f"\t\t{m.value}")
                try:
                    e = get_estimator(l, m, len(np.unique(y)), True)
                except ValueError as er:
                    print(f"\t\t\t{er}")
                    continue
                print(f"\t\t\t{e}", end="")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    e.fit(X, y)
                    e.predict(X)
                    e.predict_proba(X)
                print(f"  -  {e.score(X, y)}")


if __name__ == "__main__":
    test()
