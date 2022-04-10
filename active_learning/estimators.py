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


from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Any, Callable, Dict
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from active_learning import utils


inherintly_multiclass = {
    "BernoulliNB",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LinearDiscriminantAnalysis",
    "LinearSVC",  # (setting multi_class=”crammer_singer”)
    "LogisticRegression",  # (setting multi_class=”multinomial”)
    "LogisticRegressionCV",  # (setting multi_class=”multinomial”)
    "MLPClassifier",
    "NearestCentroid",
    "QuadraticDiscriminantAnalysis",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
}


multiclass_as_one_vs_one = {
    "NuSVC",
    "SVC",
    "GaussianProcessClassifier",  # (setting multi_class = “one_vs_one”)
}

multiclass_as_one_vs_rest = {
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",  # (setting multi_class = “one_vs_rest”)
    "LinearSVC",  # (setting multi_class=”ovr”)
    "LogisticRegression",  # (setting multi_class=”ovr”)
    "LogisticRegressionCV",  # (setting multi_class=”ovr”)
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
}


support_multilabel = {
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


support_multiclass_multioutput = {
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
}


def get_MultiOutputToMultiLabelClassifier(
    base: Callable[..., BaseEstimator]
) -> Callable[..., BaseEstimator]:
    """Get a MultiOutputToMultiLabelClassifier to wrap multioutput scikit-learn base learners.

    Parameters
    ----------
    base : Callable[..., BaseEstimator]
        A scikit-learn classifier class that is inherintly multioutput.

    Returns
    -------
    Callable[..., BaseEstimator]
        A class that inherits from base that handles the conversion of predict_proba calls.
    """

    if base.__name__ not in support_multiclass_multioutput:
        warnings.warn(
            f"WARNING: the {base.__name__} class is not inhertintly multiclass, so the"
            " MultiOutputToMultiLabelClassifier wrapper class should not be used."
        )

    class MultiOutputToMultiLabelClassifier(base):
        """Have a multioutput classifier emulate the behavior of a multilabel classifier."""

        def __init__(self, **kwargs):
            """Instantiate a scikit-learn base learner.

            Parameters
            ----------
            **kwargs
                Keyword arguments passed to the super classes's __init__ method.
            """

            super().__init__(**kwargs)
            self.predict_proba = self._predict_proba

        def __repr__(self):
            return "MultiOutputToMultiLabelClassifier(" + super().__repr__() + ")"

        def __str__(self):
            return "MultiOutputToMultiLabelClassifier(" + super().__repr__() + ")"

        def _predict_proba(self, X) -> np.ndarray:
            """Get prediction probabilities that are casted into a multilabel format.

            Parameters
            ----------
            X
                Features to make predictions upon.

            Returns
            -------
            np.ndarray
                An (n_samples, n_classes) ndarray indicating class probabilities.
            """

            probas = super().predict_proba(X)
            if not all(p.shape == (X.shape[0], 2) for p in probas):
                raise ValueError(
                    f"Expected every individual probability to have shape {(X.shape[0], 2)},"
                    f" but predict_proba() returned shape {np.array(probas).shape}."
                )
            probas = [p[:, 1] for p in probas]
            probas = np.vstack(probas).T
            return probas

        def predict(self, X) -> np.ndarray:
            """Get predictions.

            Parameters
            ----------
            X
                Features to make predictions upon.

            Returns
            -------
            np.ndarray
                An (n_samples,) ndarray indicating class predictions.
            """

            self.predict_proba = super().predict_proba
            preds = super().predict(X)
            self.predict_proba = self._predict_proba
            return preds

    return MultiOutputToMultiLabelClassifier


def get_mapper(random_state: int) -> Dict[str, Dict[str, Any]]:
    """Return the mapping of keyword arguments to BaseLearner instances.

    Parameters
    ----------
    random_state : int
        Integer for reproducible results.

    Returns
    -------
    Dict[str, BaseEstimator]
        Mapping of keyword arguments to scikit-learn learners.
    """

    mapper = {
        "ExtraTreesClassifier": {
            "cls": ExtraTreesClassifier,
            "args": [],
            "kwargs": dict(random_state=random_state),
        },
        "GradientBoostingClassifier": {
            "cls": GradientBoostingClassifier,
            "args": [],
            "kwargs": dict(random_state=random_state),
        },
        "RandomForestClassifier": {
            "cls": RandomForestClassifier,
            "args": [],
            "kwargs": dict(random_state=random_state),
        },
        "MLPClassifier": {
            "cls": MLPClassifier,
            "args": [],
            "kwargs": dict(random_state=random_state),
        },
        "SVC": {
            "cls": SVC,
            "args": [],
            "kwargs": dict(kernel="linear", probability=True, random_state=random_state),
        },
    }

    return mapper


def get_estimator(
    learner: str,
    target_type: str,
    random_state: int,
) -> BaseEstimator:
    """Get a scikit learn estimator given a desired base learner and a multiclass protocol.

    Parameters
    ----------
    learner : str
        Code to refer to the base learner to use.
    target_type : str
        Type of target vector for classification, as returned by
            sklearn.utils.multiclass.type_of_target(). Currently supported targets are one of:
            {'multilabel-indicator', 'multiclass-multioutput', 'multiclass', 'binary'}

    Returns
    -------
    BaseEstimator
        A scikit learn estimator that can be used for multiclass classification.

    Raises
    ------
    KeyError
        If the learner is not recognized.
    """

    if target_type not in {"multilabel-indicator", "multiclass", "binary"}:
        raise ValueError(f"Estimators for this kind of target: {target_type} not supported")

    mapper = get_mapper(random_state)

    if learner not in mapper:
        raise KeyError(f"Learner not recognized: {learner}")

    cls = mapper[learner]["cls"]
    args = mapper[learner]["args"]
    kwargs = mapper[learner]["kwargs"]

    clf = utils.init(cls, *args, **kwargs)()

    if target_type == "binary":
        return clf
    if target_type == "multiclass":
        if learner in inherintly_multiclass:
            return clf
        return OneVsRestClassifier(clf, n_jobs=-1)
    if target_type == "multilabel-indicator":
        if learner in support_multiclass_multioutput:
            MultiOutputToMultiLabelClassifier = get_MultiOutputToMultiLabelClassifier(cls)
            return MultiOutputToMultiLabelClassifier(*args, **kwargs)
        if learner in support_multilabel:
            return clf
        return OneVsRestClassifier(clf, n_jobs=-1)

    raise Exception("Undefined behavior occurred associated with the type of target.")
