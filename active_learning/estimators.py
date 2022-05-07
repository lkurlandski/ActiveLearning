"""Acquire a scikit-learn estimator from a string code.
"""


from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Callable
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


# Keeping this dictionary up to date can help with debugging and error messages.
valid_base_learners = {
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "RandomForestClassifier",
    "SGDClassifier",
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",
    "MLPClassifier",
    "LinearSVC",
    "SVC",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
}


# These learners are multiclass by nature and should not be wrapped in a
# multiclass wrapper.
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


# These learners are not inherintly multiclass, but support multiclass
# classification using a one-versus-one protocol.
multiclass_as_one_vs_one = {
    "NuSVC",
    "SVC",
    "GaussianProcessClassifier",  # (setting multi_class = “one_vs_one”)
}


# These learners are not inherintly multiclass, but support multiclass
# classification using a one-versus-rest protocol.
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


# These learners support multilabel classification and should not be wrapped in
# a multiclass or multilabel wrapper.
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


# These larners support multioutput classification and need their prediction
# targets modified to be compatible with ModAL.
# The MultiOutputToMultiLabelClassifier handles this.
support_multiclass_multioutput = {
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",
}


def get_multioutput_to_multilabel_wrapper(
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

    if base.__name__ not in support_multiclass_multioutput and base.__name__ != "BaseEstimator":
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
    if learner == "DecisionTreeClassifier":
        DecisionTreeClassifier_ = get_multioutput_to_multilabel_wrapper(DecisionTreeClassifier)
        clf = DecisionTreeClassifier_(random_state=random_state)
    elif learner == "ExtraTreeClassifier":
        ExtraTreeClassifier_ = get_multioutput_to_multilabel_wrapper(ExtraTreeClassifier)
        clf = ExtraTreeClassifier_(random_state=random_state)
    elif learner == "ExtraTreesClassifier":
        ExtraTreesClassifer_ = get_multioutput_to_multilabel_wrapper(ExtraTreesClassifier)
        clf = ExtraTreesClassifer_(random_state=random_state)
    elif learner == "RandomForestClassifier":
        RandomForestClassifier_ = get_multioutput_to_multilabel_wrapper(RandomForestClassifier)
        clf = RandomForestClassifier_(random_state=random_state)
    elif learner == "KNeighborsClassifier":
        KNeighborsClassifier_ = get_multioutput_to_multilabel_wrapper(KNeighborsClassifier)
        clf = KNeighborsClassifier_(random_state=random_state)
    elif learner == "RadiusNeighborsClassifier":
        RadiusNeighborsClassifier_ = get_multioutput_to_multilabel_wrapper(
            RadiusNeighborsClassifier
        )
        clf = RadiusNeighborsClassifier_(random_state=random_state)
    elif learner == "GradientBoostingClassifier":
        clf = GradientBoostingClassifier(random_state=random_state)
    elif learner == "MLPClassifier":
        clf = MLPClassifier(random_state=random_state)
    elif learner == "LinearSVC":
        clf = LinearSVC(random_state=random_state)
    elif learner == "SVC":
        clf = SVC(kernel="linear", probability=True, random_state=random_state)
    elif learner == "SGDClassifier":
        clf = SGDClassifier(random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(
            f"Learner not recognized: {learner}. "
            f"Valid learners are: {pformat(valid_base_learners)}."
        )

    if target_type == "binary":
        return clf
    if target_type == "multiclass":
        if learner in inherintly_multiclass:
            return clf
        return OneVsRestClassifier(clf, n_jobs=-1)
    if target_type == "multilabel-indicator":
        if learner in support_multilabel.union(support_multiclass_multioutput):
            return clf
        return OneVsRestClassifier(clf, n_jobs=-1)

    raise ValueError(
        f"Estimators for this kind of target: {target_type} not supported. "
        f"Valid target types are:\n{pformat({'multilabel-indicator', 'multiclass', 'binary'})}"
    )
