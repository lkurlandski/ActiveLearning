"""Acquire a scikit-learn estimator from a string code.

TODO
----
- Determine the optimal svm to use from scikit-learn.
- Implement multilabel classification.
- Expand into a module with wrappers for tensorflow, pytorch, and huggingface models.
- When doing the above, perform a big overhaul/refactor.

FIXME
-----
- Address the problem where the calibrated classifier will fail if not enough examples from each
    class exist.
"""


from dataclasses import dataclass, field
from enum import Enum
from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Set, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.svm import LinearSVC, SVC

from active_learning import utils


support_multi_label = {
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "KNeighborsClassifier",
    "MLPClassifier",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
}


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
    wrapper : Callable[..., BaseEstimator]
        A callable which resutrn a scikit-learn multiclass wrapper
    kwargs : Dict[str, Any]
        Keyword arguments to pass to the wrapper,
            defaults to a dict instructing for maximum parallelization
    """

    wrapper: Callable[..., BaseEstimator]
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
    learner : Callable[..., BaseEstimator]
        A callable which returns a scikit-learn base learner
    default_multiclass_code : MCOpts
        A string which contains the built-in multiclass support mode of the learner
    kwargs : Dict[str, Any]
        Keyword arguments to pass to the learner, defaults to an empty dict
    multiclass_options: Set[str]
        Set of valid options for the learner 'multi_class' argument, if it has one,
            defaults to an empty set
    """

    learner: Callable[..., BaseEstimator]
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
    # TODO: Investigate whether the CalibratedClassifierCV(LinearSVC()) with a Platt Scaling is
    # slower the SVC(kernel='linear')
    # Add learners alphabeically, by the name of their parent package
    mapper = {
        "RandomForestClassifier": BaseLearner(RandomForestClassifier, MCOpts.base),
        "MLPClassifier": BaseLearner(MLPClassifier, MCOpts.base, {"early_stopping": False}),
        "LinearSVC": BaseLearner(
            LinearSVC, MCOpts.ovr, multiclass_options={MCOpts.ovr, MCOpts.crammer_singer}
        ),
        "SVC": BaseLearner(
            SVC, MCOpts.ovo, {"probability": probabalistic_required, "kernel": "linear"}
        ),
    }

    return mapper


def extract_estimator(base_learner: BaseLearner) -> BaseEstimator:
    """Extract the base learner from the BaseLearner type.

    Parameters
    ----------
    base_learner : BaseLearner
        BaseLearner container from which to extract the base learner estimator from

    Returns
    -------
    BaseEstimator
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
) -> Union[BaseEstimator, OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier]:
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
    Union[BaseEstimator, OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier]
        Either the scikit learn base learner instatiated with a valid option for its multi_class
            parameter or a multiclass wrapper if requested option is invalid.
    """

    # Use the short string for this multiclass method
    if multiclass_code in base_learner.multiclass_options:
        base_learner.kwargs["multi_class"] = multiclass_code.value
        return extract_estimator(base_learner)
    # Use the alternative string for this multiclass method
    if multiclass_code in synonyms and synonyms[multiclass_code] in base_learner.multiclass_options:
        base_learner.kwargs["multi_class"] = synonyms[multiclass_code]
        return extract_estimator(base_learner)
    # Requested protocal was invalid
    return wrap_estimator_for_multiclass(base_learner, multiclass_estimator)


def get_estimator(
    base_learner_code: str,
    y: Union[np.ndarray, spmatrix],
    multiclass_code: str,
    probabalistic_required: bool = True,
) -> Union[
    BaseEstimator,
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
        BaseEstimator,
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

    target_type = type_of_target(y)
    if target_type in {"continuous", "continuous-multioutput", "multiclass-multioutput", "unknown"}:
        raise ValueError("Estimators for this kind of target not supported")

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

    multiclass_trivial = (
        target_type == "binary"
        or base_learner.default_multiclass_code == MCOpts.base
        or base_learner.default_multiclass_code == multiclass_code
    )

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

    # This functionality is currently experimental in scikit-learn and its API is subject to change
    multilabel_strs = {"multilabel", "multioutput", "multioutput_only"}
    supports_multilabel = any((v for k, v in estimator._get_tags().items() if k in multilabel_strs))
    if target_type == "multilabel-indicator" and not supports_multilabel:
        estimator = MultiOutputClassifier(estimator, n_jobs=-1)

    return estimator
